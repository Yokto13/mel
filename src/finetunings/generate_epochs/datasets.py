from pathlib import Path
import sys

from models.batch_sampler import BatchSampler

sys.stdout.reconfigure(line_buffering=True, write_through=True)

import numba as nb
import numpy as np
import numpy.typing as npt
from torch.utils.data import IterableDataset

from utils.loaders import load_embs_qids_tokens


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

        self._data_index = np.arange(len(self._embs))
        self._batch_size = batch_size
        self._max_idx = len(self._embs) // self._batch_size
        self._batch_idx = 0

    def shuffler(self):
        _rng.shuffle(self._data_index)

    def get_batch(self):
        if self._batch_idx == self._max_idx - 1:
            self._batch_idx = 0
            self.shuffler()
        indices = self._data_index[
            self._batch_idx * self._batch_size : self._batch_idx * self._batch_size
            + self._batch_size
        ]
        self._batch_idx += 1
        batch = (self._embs[indices], self._qids[indices], self._tokens[indices])
        return batch

    def __iter__(self):
        while True:
            yield self.get_batch()

    def _remove_when_qid_missing(self, data, known_qids):
        embs, qids, tokens = data
        mask = np.isin(qids, known_qids)
        return embs[mask], qids[mask], tokens[mask]


@nb.njit
def _prepare_batch(
    batch_size, line_size, per_mention, toks_size, sampler_tokens, positive, negative
):
    together_line = np.zeros((line_size, toks_size), dtype=np.int32)
    batch_Y = np.zeros((batch_size, line_size))

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


class DamuelNeighborsIterator:
    def __init__(
        self,
        batcher: Batcher,
        batch_size: int,
        neg_cnt: int,
        sampler: BatchSampler,
        sampler_tokens: npt.NDArray[np.int_],
        toks_size: int,
    ) -> None:
        self.batcher = batcher
        self.batch_size = batch_size
        self.negative_cnt = neg_cnt
        self.batch_sampler = sampler
        self.sampler_tokens = sampler_tokens
        self.toks_size = toks_size

    def __iter__(self):
        per_mention = 1 + self.negative_cnt
        line_size = per_mention * self.batch_size
        for embs, qids, toks in self.batcher:
            batch = toks

            positive, negative = self.batch_sampler.sample(
                embs, qids, self.negative_cnt
            )
            together_line, batch_Y = _prepare_batch(
                self.batch_size,
                line_size,
                per_mention,
                self.toks_size,
                self.sampler_tokens,
                positive,
                negative,
            )
            yield batch, together_line, batch_Y
