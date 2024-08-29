import logging
import os
from pathlib import Path

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp

from torch.nn.parallel import DistributedDataParallel as DDP

import wandb

from utils.argument_wrappers import ensure_datatypes
from utils.running_averages import RunningAverages

from finetunings.finetune_model.train import (
    load_model,
    LinksAndDescriptionsTogetherDataset,
    get_wandb_logs,
    SaveInformation,
    batch_recall,
    save_model,
    forward_to_embeddings,
)
from finetunings.finetune_model.ddp import setup, cleanup


# Settings ===========================================

_RUNNING_AVERAGE_SMALL = 100
_RUNNING_AVERAGE_BIG = 1000

_logger = logging.getLogger("finetuning.finetune_model.train")


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
    DATASET_DIR: Path,
    FOUNDATION_MODEL_PATH: str,
    EPOCHS: int,
    LOGIT_MULTIPLIER: int,
    LR: float,
    MODEL_SAVE_DIR: str,
    STATE_DICT_PATH: str | None,
):
    setup(rank, world_size)

    model = load_model(FOUNDATION_MODEL_PATH, STATE_DICT_PATH)
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
            },
        )

    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    running_averages = None
    if is_the_main_process:
        running_averages = RunningAverages(_RUNNING_AVERAGE_SMALL, _RUNNING_AVERAGE_BIG)

    for epoch in range(EPOCHS):
        model.train()

        train_loss = 0

        dataset = LinksAndDescriptionsTogetherDataset(DATASET_DIR, epoch)
        dataloader = DataLoader(dataset, batch_size=None, pin_memory=True)

        for together, labels in tqdm(dataloader, total=len(dataloader)):
            together = together.to(rank)

            per_replica = together.shape[0] // world_size
            replica_part = together[rank * per_replica : (rank + 1) * per_replica]

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

            outputs = torch.mm(links_embedded, descs_embedded.t())

            outputs = outputs * LOGIT_MULTIPLIER

            labels = labels.to(rank)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            loss_item = loss.item()
            train_loss += loss_item

            if is_the_main_process:
                r_at_1 = batch_recall(outputs, labels, k=1)
                r_at_10 = batch_recall(outputs, labels, k=10)

                running_averages.update_loss(loss_item)
                running_averages.update_recall(r_at_1, r_at_10)

                wand_dict = get_wandb_logs(loss_item, r_at_1, r_at_10, running_averages)
                wandb.log(
                    wand_dict,
                )

    if is_the_main_process:
        # We only save the model on the main process and only once
        # Intermediate saves could mess up synchronization
        save_information = SaveInformation(MODEL_SAVE_DIR, True)
        save_model(model.module, save_information)

    cleanup()


def cleanup():
    dist.destroy_process_group()


# Training ===========================================
@ensure_datatypes(
    [
        Path,
        str,
        int,
        int,
        float,
        str,
        str,
    ],
    {},
)
def train_ddp(
    DATASET_DIR: Path,
    FOUNDATION_MODEL_PATH: str,
    EPOCHS: int,
    LOGIT_MULTIPLIER: int,
    LR: float,
    TYPE: str = "entity_names",
    MODEL_SAVE_DIR: str = "models",
    STATE_DICT_PATH: str | None = None,
):
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
        ),
        nprocs=world_size,
    )
