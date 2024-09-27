import logging
import sys
from collections import defaultdict
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


# If we can load all the data in RAM it might be better to side step Dataloader and implement the sampling ourselves.
class Batcher:
    def __init__(self, dir_path: Path, known_qids: npt.ArrayLike, batch_size: int):
        self.dir_path = dir_path
        embs, qids, tokens = load_embs_qids_tokens(dir_path)
        self._embs, self._qids, self._tokens = self._remove_when_qid_missing(
            (embs, qids, tokens), known_qids
        )

        self._base_index = np.arange(len(self._embs))
        self._data_index = None
        self._batch_size = batch_size
        self._max_idx = None
        self._batch_idx = 0

        # Ensure data is shuffled from the start to avoid QID repetition in batches
        self._reset_index()

    def _shuffle(self) -> None:
        _rng.shuffle(self._base_index)

    def get_batch(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        indices = self._get_batch_indices()
        batch = self._construct_batch(indices)
        return batch

    def _reset_index(self) -> None:
        _logger.info("Resetting index")
        self._shuffle()
        self._data_index = self._create_unique_qid_index(
            self._base_index, self._qids, self._batch_size
        )
        self._max_idx = len(self._data_index) // self._batch_size

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

    def _construct_batch(
        self, indices: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return (self._embs[indices], self._qids[indices], self._tokens[indices])

    def _get_batch_indices(self) -> np.ndarray:
        if self._batch_idx == self._max_idx - 1:
            self._batch_idx = 0
            self._reset_index()
        indices = self._data_index[
            self._batch_idx * self._batch_size : self._batch_idx * self._batch_size
            + self._batch_size
        ]
        self._batch_idx += 1
        return indices

    def _qids_in_batch_are_unique(self, indices: np.ndarray) -> bool:
        unique_qids = np.unique(self._qids[indices])
        return len(unique_qids) == len(indices)

    def __iter__(self):
        while True:
            yield self.get_batch()

    def _remove_when_qid_missing(
        self, data: tuple[np.ndarray, np.ndarray, np.ndarray], known_qids: npt.ArrayLike
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
        batcher: Batcher,
        batch_size: int,
        neg_cnt: int,
        sampler: BatchSampler,
        sampler_tokens: npt.NDArray[np.int_],
        toks_size: int,
        return_Y: bool = True,
    ) -> None:
        self.batcher = batcher
        self.batch_size = batch_size
        self.negative_cnt = neg_cnt
        self.batch_sampler = sampler
        self.sampler_tokens = sampler_tokens
        self.toks_size = toks_size
        self.return_Y = return_Y

    def __iter__(self):
        per_mention = 1 + self.negative_cnt
        line_size = per_mention * self.batch_size
        for embs, qids, toks in self.batcher:
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
