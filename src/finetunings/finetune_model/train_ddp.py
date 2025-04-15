import logging
import os
from pathlib import Path

import numpy as np

import torch

from tqdm import tqdm

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

from utils.running_averages import RunningAverages

from finetunings.finetune_model.data import (
    LightWeightDataset,
    save_model,
    SaveInformation,
)
from finetunings.finetune_model.ddp import cleanup, setup
from finetunings.finetune_model.monitoring import process_metrics, get_gradient_norm

from finetunings.finetune_model.train import forward_to_embeddings, load_model


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


def construct_labels(dataset: LightWeightDataset) -> np.ndarray:
    labels = np.zeros((dataset.links_cnt, dataset.descriptions_cnt), dtype=np.float32)
    for i in range(dataset.links_cnt):
        r = dataset.descriptions_cnt // dataset.links_cnt
        labels[i, i * r] = 1
    return labels


@torch.compile
def _calculate_loss(
    links_embedded, descs_embedded, labels, LOGIT_MULTIPLIER, criterion
):
    outputs = torch.mm(links_embedded, descs_embedded.t())
    outputs = outputs * LOGIT_MULTIPLIER
    loss = criterion(outputs, labels) + criterion(outputs.t(), labels.t())
    return loss, outputs


def save_final_model(model, MODEL_SAVE_DIR):
    save_information = SaveInformation(MODEL_SAVE_DIR, True)
    save_model(model, save_information)


def _ddp_train(
    rank: int,
    world_size: int,
    DATASET_DIR: Path,
    FOUNDATION_MODEL_PATH: str,
    EPOCHS: int,
    LOGIT_MULTIPLIER: int,
    LR: float,
    MODEL_SAVE_DIR: str,
    STATE_DICT_PATH: str | None,
    TARGET_DIM: int | None,
    WEIGHT_DECAY: float | None,
):
    setup(rank, world_size)

    model = load_model(FOUNDATION_MODEL_PATH, STATE_DICT_PATH, TARGET_DIM)
    model = DDP(model.to(rank), device_ids=[rank])

    is_the_main_process = rank == 0

    if is_the_main_process:
        wandb.init(
            project="EL-train_ddp_process_0",
            config={
                "FOUNDATION_MODEL_PATH": FOUNDATION_MODEL_PATH,
                "EPOCHS": EPOCHS,
                "LOGIT_MULTIPLIER": LOGIT_MULTIPLIER,
                "LR": LR,
                "MODEL_SAVE_DIR": MODEL_SAVE_DIR,
                "STATE_DICT_PATH": STATE_DICT_PATH,
                "WEIGHT_DECAY": WEIGHT_DECAY,
            },
        )

    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()

    scaler = torch.amp.GradScaler("cuda")

    running_averages = None
    if is_the_main_process:
        running_averages = RunningAverages(_RUNNING_AVERAGE_SMALL, _RUNNING_AVERAGE_BIG)

    for epoch in range(EPOCHS):
        model.train()

        train_loss = 0

        dataset = LightWeightDataset(DATASET_DIR, epoch, rank, world_size)
        dataloader = DataLoader(
            dataset, batch_size=None, pin_memory=True, num_workers=2, prefetch_factor=2
        )

        labels = construct_labels(dataset)
        labels = torch.from_numpy(labels).to(rank)

        for replica_part in tqdm(dataloader, total=len(dataloader)):

            with torch.autocast(device_type="cuda"):
                replica_part = forward_to_embeddings(replica_part, model)

                with torch.no_grad():  # all_gather cannot propagate gradients so make it explicit
                    all_replicas = [
                        torch.zeros_like(replica_part) for _ in range(world_size)
                    ]
                    torch.distributed.all_gather(all_replicas, replica_part)

                # Allow gradients propagation for the slice owned by the current process
                all_replicas[rank] = replica_part

                all_replicas = torch.cat(all_replicas, dim=0)

                links_embedded, descs_embedded = (
                    all_replicas[: dataset.links_cnt],
                    all_replicas[dataset.links_cnt :],
                )

                loss, outputs = _calculate_loss(
                    links_embedded, descs_embedded, labels, LOGIT_MULTIPLIER, criterion
                )

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            norm_for_logs = get_gradient_norm(model.module)
            optimizer.zero_grad()

            loss_item = loss.item()
            train_loss += loss_item

            if is_the_main_process:
                process_metrics(
                    outputs,
                    labels,
                    loss_item,
                    running_averages,
                    {
                        "gradient_norm": norm_for_logs,
                    },
                )
        if is_the_main_process and epoch % 100 == 0:
            save_final_model(model.module, MODEL_SAVE_DIR)

    if is_the_main_process:
        # We only save the model on the main process and only once
        # Intermediate saves could mess up synchronization
        save_final_model(model.module, MODEL_SAVE_DIR)

    cleanup()


def cleanup():
    dist.destroy_process_group()


# Training ===========================================
@gin.configurable
def train_ddp(
    DATASET_DIR,
    FOUNDATION_MODEL_PATH,
    EPOCHS,
    LOGIT_MULTIPLIER,
    LR,
    MODEL_SAVE_DIR="models",
    STATE_DICT_PATH=None,
    TARGET_DIM=None,
    WEIGHT_DECAY=0.0,
):
    DATASET_DIR = Path(DATASET_DIR)
    FOUNDATION_MODEL_PATH = str(FOUNDATION_MODEL_PATH)
    EPOCHS = int(EPOCHS)
    LOGIT_MULTIPLIER = int(LOGIT_MULTIPLIER)
    LR = float(LR)
    MODEL_SAVE_DIR = str(MODEL_SAVE_DIR)
    STATE_DICT_PATH = str(STATE_DICT_PATH) if STATE_DICT_PATH is not None else None
    TARGET_DIM = int(TARGET_DIM) if TARGET_DIM is not None else None
    WEIGHT_DECAY = float(WEIGHT_DECAY)

    world_size = torch.cuda.device_count()

    mp.spawn(
        _ddp_train,
        args=(
            world_size,
            DATASET_DIR,
            FOUNDATION_MODEL_PATH,
            EPOCHS,
            LOGIT_MULTIPLIER,
            LR,
            MODEL_SAVE_DIR,
            STATE_DICT_PATH,
            TARGET_DIM,
            WEIGHT_DECAY,
        ),
        nprocs=world_size,
    )
