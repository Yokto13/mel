from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn as nn
import wandb


@dataclass
class SaveInformation:
    output_path: Path
    is_final: bool
    epoch: int = None
    recall: int = None


def _load_epoch_npz(path: Path, epoch: int | str) -> tuple:
    d = np.load(path / f"epoch_{epoch}.npz")
    return d["X"], d["lines"], d["Y"]


def _save_non_final_model(model: nn.Module, save_information: SaveInformation) -> None:
    def construct_non_final_name():
        return f"{save_information.output_path}/{wandb.run.name}_{save_information.epoch}_{save_information.recall}.pth"

    name = construct_non_final_name()
    torch.save(model.state_dict(), name)


def _save_final_model(model: nn.Module, save_information: SaveInformation) -> None:
    torch.save(model.state_dict(), f"{save_information.output_path}/final.pth")


def save_model(model: nn.Module, save_information: SaveInformation) -> None:
    if save_information.is_final:
        _save_final_model(model, save_information)
    else:
        _save_non_final_model(model, save_information)


class LinksAndDescriptionsTogetherDataset(Dataset):
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


class LightWeightDataset(Dataset):
    def __init__(
        self, dataset_dir: Path, epoch: int, rank: int = 1, word_size: int = 1
    ) -> None:
        super().__init__()
        self._word_size = word_size
        self._rank = rank
        self._dataset_dir = dataset_dir
        self._epoch = epoch
        self._data = self._load()
        self._links_cnt = None
        self._descriptions_cnt = None
        self._len = None
        self._load()

    def __len__(self):
        return self._len

    def __getitem__(self, index) -> tuple[torch.tensor, torch.tensor, torch.tensor]:
        return self._data[index]

    @property
    def links_cnt(self) -> int:
        return self._links_cnt

    @property
    def descriptions_cnt(self) -> int:
        return self._descriptions_cnt

    def _load(self) -> Any:
        self._set_cnts()
        this_share_start, this_share_end = self._get_share_bounds()
        if this_share_end <= self.links_cnt:
            # our data will be some part of the links
            data = self._load_links(this_share_start, this_share_end)
        elif this_share_start >= self.links_cnt:
            # our data will be some part of the descriptions
            data = self._load_descriptions(this_share_start, this_share_end)
        else:
            # our data are some part of both
            # the complicated case
            data = self._load_both(this_share_start, this_share_end)
        self._data = data

    def _get_share_bounds(self) -> tuple[int, int]:
        items_per_batch = self.links_cnt + self.descriptions_cnt
        assert items_per_batch % self._word_size == 0
        per_process = items_per_batch // self._word_size
        this_share_start = self._rank * per_process
        this_share_end = (self._rank + 1) * per_process
        return this_share_start, this_share_end

    def _load_links(self, start: int, end: int) -> Any:
        d = self._get_data_obj()
        return d["X"][:, start:end]

    def _load_descriptions(self, start: int, end: int) -> Any:
        d = self._get_data_obj()
        start -= self.links_cnt
        end -= self.links_cnt
        return d["lines"][:, start:end]

    def _load_both(self, start: int, end: int) -> Any:
        # No option but to load everything into memmory
        d = self._get_data_obj()
        together = np.concatenate((d["X"], d["lines"]), axis=1)
        return together[:, start:end]

    def _set_cnts(self) -> None:
        d = self._get_data_obj()
        self._links_cnt = d["X"].shape[1]
        self._descriptions_cnt = d["lines"].shape[1]
        self._len = d["X"].shape[0]

    def _get_data_obj(self) -> Any:
        d = np.load(self._dataset_dir / f"epoch_{self._epoch}.npz")
        return d
