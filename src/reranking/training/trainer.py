import logging
import os

import numpy as np
import torch

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import gin
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from finetunings.finetune_model.ddp import cleanup, setup
from reranking.training.training_configs import TrainingConfig, pairwise_mlp

# Settings ===========================================


_logger = logging.getLogger("reranking.train.trainer")


if torch.cuda.is_available():
    _logger.debug("Running on CUDA.")
    device = torch.device("cuda")
else:
    _logger.debug("CUDA is not available.")
    device = torch.device("cpu")


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


SEED = 0
torch.manual_seed(SEED)


def _ddp_train(
    rank: int,
    world_size: int,
    training_config: TrainingConfig,
    epochs,
    gradient_clip=1.0,
):
    setup(rank, world_size)

    model = training_config.model
    dataset = training_config.dataset
    optimizer = training_config.optimizer

    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)

    dataloader = DataLoader(
        dataset,
        batch_size=training_config.batch_size,
        sampler=sampler,
        pin_memory=True,
        num_workers=2,
    )

    model = DDP(model.to(rank), device_ids=[rank])
    model = torch.compile(model)

    is_the_main_process = rank == 0

    scaler = torch.amp.GradScaler("cuda")

    loss = None

    def step():
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

    step = torch.compile(step)

    for epoch in range(epochs):
        if is_the_main_process:
            _logger.info(f"Starting epoch {epoch + 1}/{epochs}")

        for links, entities, labels in dataloader:
            global_step += 1
            links = links.to(rank, non_blocking=True)
            entities = entities.to(rank, non_blocking=True)
            labels = labels.to(rank, non_blocking=True)

            batch_data = {
                "mention_tokens": links,
                "entity_tokens": entities,
                "labels": labels,
            }

            with torch.autocast(device_type="cuda"):
                loss = model.train_step(batch_data)
            step()
            if is_the_main_process and global_step % 10 == 0:
                _logger.info(f"Step {global_step}, loss: {loss.item():.4f}")

    cleanup()


def cleanup():
    dist.destroy_process_group()


# Training ===========================================
def train_ddp():
    _logger.info("Starting DDP training")
    gradient_clip = 1.0
    epochs = 10
    world_size = torch.cuda.device_count()
    _logger.debug(f"Using {world_size} GPUs for training")

    _logger.info("Loading training configuration")
    training_config = pairwise_mlp()
    _logger.info(f"Training configuration loaded: {training_config.config_name}")

    mp.spawn(
        _ddp_train,
        args=(
            world_size,
            training_config,
            epochs,
            gradient_clip,
        ),
        nprocs=world_size,
    )


if __name__ == "__main__":
    train_ddp()
