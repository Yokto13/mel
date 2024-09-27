import logging
import os
from copy import deepcopy

import numpy as np
import torch
from torch.utils.data import IterableDataset

_logger = logging.getLogger("utils.multifile_dataset")


class MultiFileDataset(IterableDataset):
    def __init__(self, data_dir, file_pattern="*.npz"):
        self.data_dir = data_dir
        self.file_pattern = file_pattern
        self.file_list = self._get_file_list()
        self._data_loader = self._choose_loader(file_pattern)

    def _get_file_list(self):
        file_list = sorted(
            [
                os.path.join(self.data_dir, f)
                for f in os.listdir(self.data_dir)
                if f.endswith(self.file_pattern[1:])
            ]
        )
        _logger.debug(str(file_list))
        return file_list

    def _load_data(self, file_path):
        return self._data_loader(file_path)

    def _choose_loader(self, file_pattern):
        match file_pattern:
            case "*.npz":
                return _npz_loader
            case _:
                raise TypeError("Unknown filepattern type")

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            for file_path in self.file_list:
                yield from self._load_data(file_path)
        else:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
            for i, file_path in enumerate(self.file_list):
                if i % num_workers == worker_id:
                    yield from self._load_data(file_path)

    def __len__(self):
        c = deepcopy(self)
        cnt = 0
        for _ in c:
            cnt += 1
        return cnt


def _npz_loader(file_path):
    d = np.load(file_path)
    for t, q in zip(d["tokens"], d["qids"]):
        yield t, q
