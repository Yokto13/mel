import logging
import os
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch

from reranking.models.pairwise_mlp import _to_device

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import gin
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
import wandb
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

from finetunings.finetune_model.data import (
    LightWeightDataset,
    SaveInformation,
    save_model,
)
from finetunings.finetune_model.ddp import cleanup, setup
from finetunings.finetune_model.monitoring import get_gradient_norm, process_metrics
from finetunings.finetune_model.train import forward_to_embeddings, load_model
from reranking.models.base import BaseRerankingModel
from reranking.models.pairwise_mlp import PairwiseMLPReranker
from utils.running_averages import RunningAverages

# Settings ===========================================

_RUNNING_AVERAGE_SMALL = 100
_RUNNING_AVERAGE_BIG = 1000

_logger = logging.getLogger("finetuning.finetune_model.train_ddp")


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
    model: BaseRerankingModel,
    dataloader,
    optimizer,
    epochs,
    gradient_clip=1.0,
):
    setup(rank, world_size)

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

        for batch in dataloader:
            global_step += 1

            with torch.autocast(device_type="cuda"):
                loss = model.train_step(_to_device(batch, rank))
            step()

    cleanup()


def cleanup():
    dist.destroy_process_group()


# Training ===========================================
@gin.configurable
def train_ddp():
    model = PairwiseMLPReranker(...)
    dataloader = ...
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    epochs = 10
    world_size = torch.cuda.device_count()

    mp.spawn(
        _ddp_train,
        args=(
            world_size,
            model,
            dataloader,
            optimizer,
            epochs,
            gradient_clip,
        ),
        nprocs=world_size,
    )
