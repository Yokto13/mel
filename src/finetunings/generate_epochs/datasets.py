from collections.abc import Iterator
from itertools import cycle
import logging
import sys
from pathlib import Path

from models.batch_sampler import BatchSampler

sys.stdout.reconfigure(line_buffering=True, write_through=True)

import numba as nb
import numpy as np
import numpy.typing as npt
from torch.utils.data import IterableDataset

from utils.loaders import load_embs_qids_tokens

_logger = logging.getLogger("finetunings.generate_epochs.datasets")
_rng = np.random.default_rng(42)


class BatcherDataset(IterableDataset):
    """
    Dataset that yields batches of data from a directory with npz link files.

    TODO: The fact that we create batches like this might not be optimal.
    Historically we were able to load all data into memory and not using DataLoader was fast.
    Right now, I am not so sure about that and we should profile it.
    DataLoader would give us also some features that are currently hard to implement here, like concurrent data loading.
    My guess is that a lot of the time is spent just on decoding the compressed numpy files.

    Args:
        dir_path: Path to the directory with npz link files.
        known_qids: Qids that should be present in the dataset. It might happen that some links does not
        have a counterpart in the set of descriptions, these links are removed.
        batch_size: The number of links in a batch.
    """

    def __init__(self, dir_path: Path, known_qids: npt.ArrayLike, batch_size: int):

        self.dir_path = dir_path
        self.known_qids = known_qids
        self.batch_size = batch_size
        self.file_paths = cycle(dir_path.glob("*.npz"))

    def __iter__(self) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray]]:
        for file_path in self.file_paths:
            embs, qids, tokens = load_embs_qids_tokens(file_path)

            embs, qids, tokens = self._remove_when_qid_missing(
                (embs, qids, tokens), self.known_qids
            )

            p = _rng.permutation(len(embs))
            embs, qids, tokens = embs[p], qids[p], tokens[p]

            base_index = np.arange(len(embs))
            data_index = self._create_unique_qid_index(
                base_index, qids, self.batch_size
            )
            max_idx = len(data_index) // self.batch_size

            for batch_idx in range(max_idx):
                indices = data_index[
                    batch_idx * self.batch_size : (batch_idx + 1) * self.batch_size
                ]
                batch = self._construct_batch((embs, qids, tokens), indices)
                yield batch

    @staticmethod
    @nb.njit
    def _create_unique_qid_index(
        base_index: np.ndarray, qids: np.ndarray, batch_size: int
    ) -> np.ndarray:
        data_idx = np.empty(len(base_index), dtype=np.int64)
        qids_in_batch = set()
        idx_counter = 0
        for idx in base_index:
            qid = qids[idx]
            if qid not in qids_in_batch:
                data_idx[idx_counter] = idx
                idx_counter += 1
                qids_in_batch.add(qid)
                if len(qids_in_batch) == batch_size:
                    qids_in_batch.clear()
        return data_idx[:idx_counter]

    @staticmethod
    def _construct_batch(
        data: tuple[np.ndarray, np.ndarray, np.ndarray], indices: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        embs, qids, tokens = data
        return embs[indices], qids[indices], tokens[indices]

    @staticmethod
    def _remove_when_qid_missing(
        data: tuple[np.ndarray, np.ndarray, np.ndarray], known_qids: npt.ArrayLike
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        embs, qids, tokens = data
        mask = np.isin(qids, known_qids)
        return embs[mask], qids[mask], tokens[mask]


@nb.njit
def prepare_batch(batch_size, line_size, toks_size, sampler_tokens, positive, negative):
    together_line = np.empty((line_size, toks_size), dtype=np.int32)

    together_line_idx = 0

    # If index error check that we are not at the end of data
    # if so then len(embs) < batch_size is likely to happen.
    for i in range(batch_size):
        pos_idx, neg_ids = positive[i], negative[i]

        together_line[together_line_idx] = sampler_tokens[pos_idx]
        together_line_idx += 1

        together_line[together_line_idx : together_line_idx + len(neg_ids)] = (
            sampler_tokens[neg_ids]
        )
        together_line_idx += len(neg_ids)

    return together_line


class DamuelNeighborsIterator:
    def __init__(
        self,
        batcher_dataset: BatcherDataset,
        batch_size: int,
        neg_cnt: int,
        sampler: BatchSampler,
        sampler_tokens: npt.NDArray[np.int_],
        toks_size: int,
    ) -> None:
        self.batcher_dataset = batcher_dataset
        self.batch_size = batch_size
        self.negative_cnt = neg_cnt
        self.batch_sampler = sampler
        self.sampler_tokens = sampler_tokens
        self.toks_size = toks_size

    def __iter__(self):
        per_mention = 1 + self.negative_cnt
        line_size = per_mention * self.batch_size
        for embs, qids, toks in self.batcher_dataset:
            batch = toks

            positive, negative = self.batch_sampler.sample(
                embs, qids, self.negative_cnt
            )
            together_line = prepare_batch(
                self.batch_size,
                line_size,
                self.toks_size,
                self.sampler_tokens,
                positive,
                negative,
            )
            yield batch, together_line
