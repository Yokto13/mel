from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Any

import numpy as np
from fire import Fire
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

import wandb

from utils.argument_wrappers import ensure_datatypes
from utils.running_averages import RunningAverages
from utils.embeddings import create_attention_mask
from utils.model_factory import ModelFactory

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


@dataclass
class _SaveInformation:
    type: str
    output_path: Path
    is_final: bool
    epoch: int = None
    recall: int = None


def _load_epoch_npz(path: Path, epoch: int | str) -> tuple:
    d = np.load(path / f"epoch_{epoch}.npz")
    return d["X"], d["lines"], d["Y"]


def _batch_recall(outputs: torch.tensor, target: torch.tensor, k: int = 1) -> float:
    """Calculates recall inside the batch.

    The calculation is done per each row. The exact values of outputs and target are not importand only orderings matter.
    Consequently this works both with logits and softmax. If k is greater than the number of classes **returns 0**.

    Args:
        outputs (torch.tensor): Matrix where each row corresponds to one multiclass classification.
        target (torch.tensor): Matrix where each row corresponds to one multiclass classification, same shape as outputs.
        k (int, optional): Recall at K. Defaults to 1.

    Returns:
        float: Recall at K for this batch.
    """
    if len(outputs[0]) < k:  # batch is too small.
        return 0.0
    _, top_indices = outputs.topk(k, dim=-1)
    top_values = target.gather(-1, top_indices)
    recall_per_row = top_values.any(dim=-1).float()
    return recall_per_row.mean().item()


def _save_non_final_model(model: nn.Module, save_information: _SaveInformation) -> None:
    def construct_non_final_name():
        return f"{save_information.output_path}/{wandb.run.name}_{save_information.epoch}_{save_information.recall}.pth"

    name = construct_non_final_name()
    torch.save(model.state_dict(), name)


def _save_final_model(model: nn.Module, save_information: _SaveInformation) -> None:
    torch.save(model.state_dict(), f"{save_information.output_path}/final.pth")


def _save_model(model: nn.Module, save_information: _SaveInformation) -> None:
    if save_information.is_final:
        _save_final_model(model, save_information)
    else:
        _save_non_final_model(model, save_information)


def _forward_to_embeddings(toks: torch.tensor, model: nn.ModuleDict) -> torch.tensor:
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


def _get_wandb_logs(
    loss_item: float, r_at_1: float, r_at_10: float, running_averages: RunningAverages
) -> dict:
    return {
        "loss": loss_item,
        "r_at_1": r_at_1,
        "r_at_10": r_at_10,
        "running_loss": running_averages.loss,
        "running_r_at_1": running_averages.recall_1,
        "running_r_at_10": running_averages.recall_10,
        "running_loss_big": running_averages.loss_big,
        "running_r_at_1_big": running_averages.recall_1_big,
        "running_r_at_10_big": running_averages.recall_10_big,
    }


class _LinksAndDescriptionsTogetherDataset(Dataset):
    def __init__(self, dataset_dir: Path, epoch: int) -> None:
        super().__init__()
        self._links, self._descriptions, self._Y = _load_epoch_npz(dataset_dir, epoch)

    def __len__(self):
        return self._links.shape[0]

    def __getitem__(self, index) -> tuple[torch.tensor, torch.tensor, torch.tensor]:
        links = self._links[index]
        descriptions = self._descriptions[index]
        y = self._Y[index]
        together = np.concatenate((links, descriptions))

        return together, y

    @property
    def links_cnt(self) -> int:
        return self._links.shape[1]

    @property
    def descriptions_cnt(self) -> int:
        return self._descriptions.shape[1]


def _load_model(model_path: str, state_dict_path: str | None) -> nn.Module:
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
    TYPE: str = "entity_names",
    MODEL_SAVE_DIR: str = "models",
    STATE_DICT_PATH: str | None = None,
):
    model = _load_model(FOUNDATION_MODEL_PATH, STATE_DICT_PATH)
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
        dataset = _LinksAndDescriptionsTogetherDataset(DATASET_DIR, epoch)
        dataloader = DataLoader(
            dataset, batch_size=None, num_workers=4, pin_memory=True
        )

        for i, (together, labels) in enumerate(tqdm(dataloader, total=len(dataset))):
            # assert i <= 100

            together = together.to(device)

            together = _forward_to_embeddings(together, model)

            links_embedded, descs_embedded = (
                together[: dataset.links_cnt],
                together[dataset.links_cnt :],
            )

            outputs = torch.mm(links_embedded, descs_embedded.t())

            outputs = outputs * LOGIT_MULTIPLIER

            labels = labels.to(device)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            loss_item = loss.item()
            train_loss += loss_item

            r_at_1 = _batch_recall(outputs, labels, k=1)
            r_at_10 = _batch_recall(outputs, labels, k=10)

            running_averages.update_loss(loss_item)
            running_averages.update_recall(r_at_1, r_at_10)

            wand_dict = _get_wandb_logs(loss_item, r_at_1, r_at_10, running_averages)
            wandb.log(
                wand_dict,
            )
        _logger.debug(f"Train loss: {train_loss / len(dataset)}")

        model.to("cpu")
        if epoch % 50 == 0:
            save_information = _SaveInformation(
                TYPE,
                MODEL_SAVE_DIR,
                False,
                epoch,
                wand_dict["running_r_at_1_big"],
            )
            _save_model(model.module, save_information)

    save_information = _SaveInformation(TYPE, MODEL_SAVE_DIR, True)
    _save_model(model.module, save_information)


if __name__ == "__main__":
    Fire(train)
