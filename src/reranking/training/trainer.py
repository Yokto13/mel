import logging
import os
from itertools import islice

import torch

import wandb
from reranking.models.pairwise_mlp import PairwiseMLPReranker
from tests.utils.test_embeddings import model

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from reranking.training.training_configs import (
    TrainingConfig,
    get_config_from_name,
)

# Settings ===========================================


_logger = logging.getLogger("reranking.train.trainer")


if torch.cuda.is_available():
    _logger.debug("Running on CUDA.")
    device = torch.device("cuda")
else:
    _logger.debug("CUDA is not available.")
    device = torch.device("cpu")


def setup(rank, world_size, master_port: str = "12355"):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = master_port

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


SEED = 0
torch.manual_seed(SEED)


def _ddp_train(
    rank: int,
    world_size: int,
    config_name: str,
    gradient_clip=1.0,
    master_port: str = "12355",
):
    setup(rank, world_size, master_port)

    is_the_main_process = rank == 0

    _logger.info("Loading training configuration")
    training_config = get_config_from_name(config_name)
    _logger.info(f"Training configuration loaded: {training_config.config_name}")

    if is_the_main_process:
        wandb.init(
            project="EL-reranking_train_ddp_process_0",
            config={
                "config_name": training_config.config_name,
                "batch_size": training_config.batch_size,
                "save_each": training_config.save_each,
                "validate_each": training_config.validate_each,
                "validation_size": training_config.validation_size,
                "output_dir": training_config.output_dir,
                "epochs": training_config.epochs,
            },
        )

    model = training_config.model
    dataset = training_config.dataset
    optimizer = training_config.optimizer

    dataloader = DataLoader(
        dataset,
        batch_size=training_config.batch_size,
        pin_memory=True,
        num_workers=4,
    )

    num_validation_batches = training_config.validation_size // training_config.batch_size
    validation_batches = list(islice(iter(dataloader), num_validation_batches))
    val_dataloader = validation_batches

    model.to(rank)
    model = torch.compile(model)
    model.model = DDP(model.model, device_ids=[rank])

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

    for epoch in range(training_config.epochs):
        if is_the_main_process:
            _logger.info(f"Starting epoch {epoch + 1}/{training_config.epochs}")

        for links, entities, labels, qids in dataloader:
            global_step += 1
            links = links.to(rank, non_blocking=True)
            entities = entities.to(rank, non_blocking=True)
            labels = labels.to(rank, non_blocking=True)
            qids = qids.to(rank, non_blocking=True)

            batch_data = {
                "mention_tokens": links,
                "entity_tokens": entities,
                "labels": labels,
                "qids": qids,
            }

            with torch.autocast(device_type="cuda"):
                loss = model.train_step(batch_data)
            step(loss)
            if is_the_main_process and global_step % training_config.save_each == 0:
                path = training_config.get_output_path(global_step)
                _logger.info(f"Saving model at step {global_step} to {path}")
                model.save(path)
            if is_the_main_process and global_step % 500 == 0:
                wandb.log({"train/loss": loss.item()}, step=global_step)
                _logger.info(f"Step {global_step}, loss: {loss.item():.4f}")
            if is_the_main_process and global_step % training_config.validate_each == 0:
                model.eval()

                correct = 0
                total = 0

                total_loss = 0.0
                val_steps = 0
                for links, entities, labels, qids in val_dataloader:
                    links = links.to(rank, non_blocking=True)
                    entities = entities.to(rank, non_blocking=True)
                    labels = labels.to(rank, non_blocking=True)
                    qids = qids.to(rank, non_blocking=True)

                    val_steps += 1

                    with torch.inference_mode():
                        if (
                            training_config.config_name == "pairwise_mlp"
                            or training_config.config_name == "pairwise_mlp_debug"
                            or training_config.config_name == "full_lealla"
                        ):
                            probs = model.score_from_tokens(links, entities)
                        else:
                            probs = model.score_from_tokens(links, qids)
                        loss = torch.nn.functional.binary_cross_entropy(probs, labels.float())
                        total_loss += loss.item()
                        predictions = (probs > 0.5).long()
                        correct += (predictions == labels).sum().item()
                        total += labels.size(0)
                if is_the_main_process:
                    _logger.info(f"Validation loss: {total_loss / val_steps:.4f}")
                    _logger.info(f"Validation accuracy: {correct / total:.4f}")

                model.train()

        if is_the_main_process:
            _logger.info(f"Epoch {epoch + 1} finished.")
    model.save(training_config.get_output_path(global_step))

    cleanup()


# Training ===========================================
def train_ddp(config_name: str, master_port: str = "12355"):
    _logger.info("Starting DDP training")
    gradient_clip = 1.0
    world_size = torch.cuda.device_count()
    _logger.debug(f"Using {world_size} GPUs for training")

    mp.spawn(
        _ddp_train,
        args=(
            world_size,
            config_name,
            gradient_clip,
            master_port,
        ),
        nprocs=world_size,
    )


if __name__ == "__main__":
    train_ddp("pairwise_mlp")
