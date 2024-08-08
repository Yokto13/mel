from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any
import numpy as np

from utils.embeddings import create_attention_mask

sys.stdout.reconfigure(line_buffering=True, write_through=True)

from fire import Fire
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from transformers import BertModel

import wandb

from utils.argument_wrappers import ensure_datatypes
from utils.running_averages import RunningAverages

# Settings ===========================================

_RUNNING_AVERAGE_SMALL = 100
_RUNNING_AVERAGE_BIG = 1000


if torch.cuda.is_available():
    print("CUDA is available!")
    device = torch.device("cuda")
else:
    print("CUDA is not available.")
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


def load_epoch_npz(path, epoch):
    d = np.load(path / f"epoch_{epoch}.npz")
    return d["X"], d["lines"], d["Y"]


def batch_recall(outputs, target, k=1):
    _, top_indices = outputs.topk(k, dim=-1)
    top_values = target.gather(-1, top_indices)
    recall_per_row = top_values.any(dim=-1).float()
    return recall_per_row.mean()


def save_non_final_model(model, save_information: _SaveInformation):
    def construct_non_final_name():
        return f"{save_information.output_path}/{wandb.run.name}_{save_information.epoch}_{save_information.recall}.pth"

    name = construct_non_final_name()
    torch.save(model.state_dict(), name)


def save_final_model(model, save_information: _SaveInformation):
    torch.save(model.state_dict(), f"{save_information.output_path}/final.pth")


def save_model(wrapper, save_information: _SaveInformation):
    if save_information.is_final:
        save_final_model(wrapper, save_information)
    else:
        save_non_final_model(wrapper, save_information)


def forward_to_embeddings(toks, model):
    att = create_attention_mask(toks)
    embeddings = model(toks, att).pooler_output
    return torch.nn.functional.normalize(embeddings, p=2, dim=1)


class _SplitToTwoDataset(Dataset):
    def __init__(self, dataset_dir: Path, epoch: int) -> None:
        super().__init__()
        self._links, self._descriptions, self._Y = load_epoch_npz(dataset_dir, epoch)

    def __len__(self):
        return self._links.shape[0]

    def __getitem__(self, index) -> Any:
        links = self._links[index]
        descriptions = self._descriptions[index]
        y = self._Y[index]
        mid_point_in_descriptions = (
            self.descriptions_cnt + self.links_cnt
        ) // 2 - self.links_cnt

        first_half = np.concatenate((links, descriptions[:mid_point_in_descriptions]))
        second_half = descriptions[mid_point_in_descriptions:]

        return first_half, second_half, y

    @property
    def links_cnt(self):
        return self._links.shape[1]

    @property
    def descriptions_cnt(self):
        return self._descriptions.shape[1]


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
        # Path,
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
):
    # print all params
    print("BATCH_DIR:", DATASET_DIR)
    print("MODEL_NAME:", FOUNDATION_MODEL_PATH)
    print("EPOCHS:", EPOCHS)
    print("LOGIT_MULTIPLIER:", LOGIT_MULTIPLIER)
    print("TYPE:", TYPE)
    print("LR:", LR)

    model = BertModel.from_pretrained(FOUNDATION_MODEL_PATH)

    model = nn.DataParallel(model)

    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=LR)

    running_averages = RunningAverages(_RUNNING_AVERAGE_SMALL, _RUNNING_AVERAGE_BIG)

    criterion = nn.CrossEntropyLoss()

    print("Starting training")
    model.to(device)
    model.train()
    for epoch in range(EPOCHS):

        train_loss = 0

        print("EPOCH:", epoch)
        split_two_dataset = _SplitToTwoDataset(DATASET_DIR, epoch)
        dataloader = DataLoader(
            split_two_dataset, batch_size=None, num_workers=4, pin_memory=True
        )

        for first_half, second_half, labels in tqdm(
            dataloader, total=len(split_two_dataset)
        ):

            first_half = first_half.to(device)

            # let's keep first_half on the GPU because the memore overhead seems to be mostly in the optimizer.
            first_half = forward_to_embeddings(first_half, model)

            second_half = second_half.to(device)
            second_half = forward_to_embeddings(second_half, model)

            links_embedded = first_half[: split_two_dataset.links_cnt]
            descs_embedded = torch.cat(
                (first_half[split_two_dataset.links_cnt :], second_half)
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

            r_at_1 = batch_recall(outputs, labels, k=1)
            r_at_10 = torch.tensor(0)
            if len(outputs[0]) >= 10:  # if batch is too small, we can't calculate r@10
                r_at_10 = batch_recall(outputs, labels, k=10)

            running_averages.update_loss(loss_item)
            running_averages.update_recall(r_at_1.item(), r_at_10.item())

            wand_dict = {
                "loss": loss_item,
                "r_at_1": r_at_1.item(),
                "r_at_10": r_at_10.item(),
                "running_loss": running_averages.loss,
                "running_r_at_1": running_averages.recall_1,
                "running_r_at_10": running_averages.recall_10,
                "running_loss_big": running_averages.loss_big,
                "running_r_at_1_big": running_averages.recall_1_big,
                "running_r_at_10_big": running_averages.recall_10_big,
            }
            wandb.log(
                wand_dict,
            )
        print(f"Train loss: {train_loss / len(split_two_dataset)}")

        model.to("cpu")
        if epoch % 50 == 0:
            save_information = _SaveInformation(
                TYPE,
                MODEL_SAVE_DIR,
                False,
                epoch,
                wand_dict["running_r_at_1_big"],
            )
            save_model(model, save_information)

    save_information = _SaveInformation(TYPE, MODEL_SAVE_DIR, True)
    save_model(model, save_information)


if __name__ == "__main__":
    Fire(train)
