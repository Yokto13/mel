import logging
from pathlib import Path

from fire import Fire
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb

from utils.argument_wrappers import ensure_datatypes
from utils.running_averages import RunningAverages
from utils.embeddings import create_attention_mask
from utils.model_factory import ModelFactory
from finetunings.finetune_model.monitoring import get_wandb_logs, batch_recall
from finetunings.finetune_model.data import (
    LinksAndDescriptionsTogetherDataset,
    SaveInformation,
    save_model,
)

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


def forward_to_embeddings(toks: torch.tensor, model: nn.ModuleDict) -> torch.tensor:
    """Calculates normalized embeddings. Attentions are created automatically. Assumes 0 is the padding token.

    Args:
        toks (torch.tensor): tokens
        model (nn.ModuleDict): model

    Returns:
        torch.tensor: normalized embs.
    """
    att = create_attention_mask(toks)
    embeddings = model(toks, att).pooler_output
    return torch.nn.functional.normalize(embeddings, p=2, dim=1)


def load_model(model_path: str, state_dict_path: str | None) -> nn.Module:
    if state_dict_path is None:
        return ModelFactory.load_bert_from_file(model_path)
    else:
        return ModelFactory.load_bert_from_file_and_state_dict(
            model_path, state_dict_path
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

            wand_dict = get_wandb_logs(loss_item, r_at_1, r_at_10, running_averages)
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
