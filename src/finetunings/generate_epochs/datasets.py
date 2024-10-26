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


# Might be usefull when we get to the point where Batcher cannot fit to memory.
class TokensIterableDataset(IterableDataset):
    def __init__(self, dir_path: Path, known_qids: set):
        self.dir_path = dir_path
        self.embs, self.qids, self.tokens = load_embs_qids_tokens(dir_path)
        self.known_qids = known_qids

    def __iter__(self):
        for embs, qid, tok in zip(self.embs, self.qids, self.tokens):
            if qid not in self.known_qids:
                continue
            yield embs, qid, tok


_rng = np.random.default_rng(42)


class BatcherDataset(IterableDataset):
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

            p = np.random.permutation(len(embs))
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
def _prepare_batch_with_Y(
    batch_size, line_size, per_mention, toks_size, sampler_tokens, positive, negative
):
    together_line = np.empty((line_size, toks_size), dtype=np.int32)
    batch_Y = np.empty((batch_size, line_size))

    together_line_idx = 0

    # If index error check that we are not at the end of data
    # if so then len(embs) < batch_size is likely to happen.
    for i in range(batch_size):
        pos_idx, neg_ids = positive[i], negative[i]

        batch_Y[i, i * per_mention] = 1

        together_line[together_line_idx] = sampler_tokens[pos_idx]
        together_line_idx += 1

        together_line[together_line_idx : together_line_idx + len(neg_ids)] = (
            sampler_tokens[neg_ids]
        )
        together_line_idx += len(neg_ids)

    return together_line, batch_Y


@nb.njit
def _prepare_batch(
    batch_size, line_size, toks_size, sampler_tokens, positive, negative
):
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
        return_Y: bool = True,
    ) -> None:
        self.batcher_dataset = batcher_dataset
        self.batch_size = batch_size
        self.negative_cnt = neg_cnt
        self.batch_sampler = sampler
        self.sampler_tokens = sampler_tokens
        self.toks_size = toks_size
        self.return_Y = return_Y

    def __iter__(self):
        per_mention = 1 + self.negative_cnt
        line_size = per_mention * self.batch_size
        for embs, qids, toks in self.batcher_dataset:
            batch = toks

            positive, negative = self.batch_sampler.sample(
                embs, qids, self.negative_cnt
            )
            if self.return_Y:
                together_line, batch_Y = _prepare_batch_with_Y(
                    self.batch_size,
                    line_size,
                    per_mention,
                    self.toks_size,
                    self.sampler_tokens,
                    positive,
                    negative,
                )
                yield batch, together_line, batch_Y
            else:
                together_line = _prepare_batch(
                    self.batch_size,
                    line_size,
                    self.toks_size,
                    self.sampler_tokens,
                    positive,
                    negative,
                )
                yield batch, together_line
