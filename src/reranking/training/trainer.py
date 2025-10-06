import copy
import logging
import os

import numpy as np
import torch

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

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

    validation_dataset = torch.utils.data.Subset(
        dataset,
        indices=np.arange(training_config.validation_size),
    )
    train_dataset = torch.utils.data.Subset(
        dataset,
        indices=np.arange(training_config.validation_size, len(dataset)),
    )

    sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)

    val_dataloader = DataLoader(
        validation_dataset,
        batch_size=training_config.batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=2,
    )

    copied_model = copy.deepcopy(model)

    model.to(rank)
    model.model = DDP(model.model, device_ids=[rank])
    model = torch.compile(model)

    is_the_main_process = rank == 0

    scaler = torch.amp.GradScaler("cuda")

    @torch.compile
    def step(current_loss):
        scaler.scale(current_loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

    global_step = 0

    for epoch in range(epochs):
        if is_the_main_process:
            _logger.info(f"Starting epoch {epoch + 1}/{epochs}")

        # Ensure proper shuffling across epochs with DistributedSampler
        sampler.set_epoch(epoch)

        dataloader = DataLoader(
            train_dataset,
            batch_size=training_config.batch_size,
            sampler=sampler,
            pin_memory=True,
            num_workers=2,
        )

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
            step(loss)
            if is_the_main_process and global_step % training_config.save_each == 0:
                path = training_config.get_output_path(global_step)
                _logger.info(f"Saving model at step {global_step} to {path}")
                copied_model.model = model.model.module
                torch.save(copied_model.state_dict(), path)
            if is_the_main_process and global_step % 500 == 0:
                _logger.info(f"Step {global_step}, loss: {loss.item():.4f}")
            if is_the_main_process and global_step % training_config.validate_each == 0:
                model.eval()

                correct = 0
                total = 0

                total_loss = 0.0
                val_steps = 0
                for links, entities, labels in val_dataloader:
                    links = links.to(rank, non_blocking=True)
                    entities = entities.to(rank, non_blocking=True)
                    labels = labels.to(rank, non_blocking=True)

                    val_steps += 1

                    with torch.inference_mode():
                        probs = model.score(links, entities).view(-1)
                        loss = torch.nn.functional.binary_cross_entropy(probs, labels.float())
                        total_loss += loss.item()
                        predictions = (torch.sigmoid(probs) > 0.5).long()
                        correct += (predictions == labels).sum().item()
                        total += labels.size(0)
                if is_the_main_process:
                    _logger.info(f"Validation loss: {total_loss / val_steps:.4f}")
                    _logger.info(f"Validation accuracy: {correct / total:.4f}")

                model.train()

        if is_the_main_process:
            _logger.info(f"Epoch {epoch + 1} finished.")

    cleanup()


def get_config_from_name(config_name: str) -> TrainingConfig:
    if config_name == "pairwise_mlp":
        return pairwise_mlp()
    else:
        raise ValueError(f"Unknown training configuration: {config_name}")


# Training ===========================================
def train_ddp(config_name: str):
    _logger.info("Starting DDP training")
    gradient_clip = 1.0
    epochs = 10
    world_size = torch.cuda.device_count()
    _logger.debug(f"Using {world_size} GPUs for training")

    _logger.info("Loading training configuration")
    training_config = get_config_from_name(config_name)
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
    train_ddp("pairwise_mlp")
