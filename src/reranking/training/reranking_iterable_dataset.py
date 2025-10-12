"""Minimal iterable dataset for streaming reranking training data."""

from pathlib import Path
from typing import Iterator, List, Tuple

import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info


class RerankingIterableDataset(IterableDataset):
    """Yield ``(link_tokens, description_tokens, labels, qids)`` from NPZ files.

    Each worker spawned by ``DataLoader`` receives a disjoint stride of the
    underlying NPZ shards, ensuring the samples are not duplicated when
    ``num_workers > 0``.
    """

    def __init__(
        self,
        data_dir: str | Path = "~/troja/outputs/reranking_test/reranker_dataset_with_qids",
    ) -> None:
        super().__init__()
        self.data_dir = Path(data_dir).expanduser()
        if not self.data_dir.is_dir():
            raise FileNotFoundError(f"Dataset directory not found: {self.data_dir}")

        self._files: List[Path] = sorted(self.data_dir.glob("*.npz"))
        if not self._files:
            raise FileNotFoundError(
                f"No NPZ files found in {self.data_dir}; expected reranking shards"
            )

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        worker_info = get_worker_info()
        if worker_info is None:
            worker_id = 0
            num_workers = 1
        else:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers

        file_paths = self._files[worker_id::num_workers]

        for file_path in file_paths:
            with np.load(file_path, allow_pickle=False) as data:
                qids = torch.from_numpy(data["qids"]).long()
                labels = torch.from_numpy(data["y"]).float()
                description_tokens = torch.from_numpy(data["description_tokens"]).long()
                link_tokens = torch.from_numpy(data["link_tokens"]).long()

            if not (len(qids) == len(labels) == len(description_tokens) == len(link_tokens)):
                raise ValueError(
                    "Mismatched array lengths in NPZ file "
                    f"{file_path}: qids={len(qids)} labels={len(labels)} "
                    f"description_tokens={len(description_tokens)} link_tokens={len(link_tokens)}"
                )

            permutation = torch.randperm(len(qids))
            qids = qids[permutation]
            labels = labels[permutation]
            description_tokens = description_tokens[permutation]
            link_tokens = link_tokens[permutation]

            for idx in range(len(qids)):
                yield (
                    link_tokens[idx],
                    description_tokens[idx],
                    labels[idx],
                    qids[idx],
                )
