import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

import wandb

from fire import Fire
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.argument_wrappers import ensure_datatypes
from utils.embeddings import create_attention_mask
from utils.model_factory import ModelFactory
from utils.running_averages import RunningAverages

from finetunings.finetune_model.data import (
    LinksAndDescriptionsTogetherDataset,
    save_model,
    SaveInformation,
)
from finetunings.finetune_model.monitoring import batch_recall, _get_wandb_logs

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

SEED = 0
torch.manual_seed(SEED)


@torch.compile
def forward_to_embeddings(toks: torch.tensor, model: nn.ModuleDict) -> torch.tensor:
    """Calculates normalized embeddings. Attentions are created automatically. Assumes 0 is the padding token.

    Args:
        toks (torch.tensor): tokens
        model (nn.ModuleDict): model

    Returns:
        torch.tensor: normalized embs.
    """
    att = create_attention_mask(toks)
    embeddings = model(toks, att)
    return torch.nn.functional.normalize(embeddings, p=2, dim=1)


def load_model(
    model_path: str,
    state_dict_path: str | None,
    target_dim: int | None,
    output_type: str | None = None,
) -> nn.Module:
    return ModelFactory.auto_load_from_file(
        model_path, state_dict_path, target_dim, output_type
    )


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
def train(
    DATASET_DIR: Path,
    FOUNDATION_MODEL_PATH: str,
    EPOCHS: int,
    LOGIT_MULTIPLIER: int,
    LR: float,
    MODEL_SAVE_DIR: str = "models",
    STATE_DICT_PATH: str | None = None,
):
    print("STATEDICTPATH", STATE_DICT_PATH)
    model = load_model(FOUNDATION_MODEL_PATH, STATE_DICT_PATH)
    model = nn.DataParallel(model)
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=LR)

    running_averages = RunningAverages(_RUNNING_AVERAGE_SMALL, _RUNNING_AVERAGE_BIG)

    criterion = nn.CrossEntropyLoss()

    _logger.debug("Starting training")
    for epoch in range(EPOCHS):
        model.to(device)
        model.train()

        train_loss = 0

        _logger.debug(f"EPOCH: {epoch}")
        dataset = LinksAndDescriptionsTogetherDataset(DATASET_DIR, epoch)
        dataloader = DataLoader(
            dataset, batch_size=None, num_workers=2, pin_memory=True
        )

        for i, (together, labels) in enumerate(tqdm(dataloader, total=len(dataset))):
            # assert i <= 100

            together = together.to(device)

            together = forward_to_embeddings(together, model)

            links_embedded, descs_embedded = (
                together[: dataset.links_cnt],
                together[dataset.links_cnt :],
            )

            # links_embed (bs, dim)
            # descs_embed (bs * (1 + neg), dim)
            # outputs (bs, bs * (1 + neg))
            outputs = torch.mm(links_embedded, descs_embedded.t())

            outputs = outputs * LOGIT_MULTIPLIER

            labels = labels.to(device)
            loss = criterion(outputs, labels)
            # loss = criterion(outputs, labels) + criterion(outputs.t(), labels.t())

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            loss_item = loss.item()
            train_loss += loss_item

            r_at_1 = batch_recall(outputs, labels, k=1)
            r_at_10 = batch_recall(outputs, labels, k=10)

            running_averages.update_loss(loss_item)
            running_averages.update_recall(r_at_1, r_at_10)

            wand_dict = _get_wandb_logs(loss_item, r_at_1, r_at_10, running_averages)
            wandb.log(
                wand_dict,
            )
        _logger.info(f"Train loss: {train_loss / len(dataset)}")

        model.to("cpu")
        if epoch % 50 == 0:
            save_information = SaveInformation(
                MODEL_SAVE_DIR,
                False,
                epoch,
                wand_dict["running_r_at_1_big"],
            )
            save_model(model.module, save_information)

    save_information = SaveInformation(MODEL_SAVE_DIR, True)
    save_model(model.module, save_information)


if __name__ == "__main__":
    Fire(train)
