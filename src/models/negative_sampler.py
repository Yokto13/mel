from random import shuffle
import numba
import numpy as np

from models.searchers.searcher import Searcher


def _sample_shuffling_numba(batch_qids, negative_cnts, neighbors, neighbors_mask):
    res = np.empty((len(batch_qids), negative_cnts), dtype=np.int32)
    for i in range(len(batch_qids)):
        ns = neighbors[i][neighbors_mask[i]]
        np.random.shuffle(ns)
        res[i] = ns[:negative_cnts]
    return res


@numba.njit
def _sample_top_numba(batch_qids, negative_cnts, neighbors, neighbors_mask):
    res = np.empty((len(batch_qids), negative_cnts), dtype=np.int32)
    for i in range(len(batch_qids)):
        ns = neighbors[i][neighbors_mask[i]]
        res[i] = ns[:negative_cnts]
    return res


def _get_sampler(sampler_type: str) -> callable:
    if sampler_type == "shuffle":
        return _sample_shuffling_numba
    if sampler_type == "top":
        return _sample_top_numba


class NegativeSampler:
    def __init__(
        self,
        embs: np.ndarray,
        qids: np.ndarray,
        searcher_constructor: type[Searcher],
        sampling_type: str,
    ) -> None:
        assert len(embs) == len(qids)
        self.embs = embs
        self.qids = qids
        self.searcher = searcher_constructor(embs, np.arange(len(embs)))
        print(sampling_type)
        self.sample_f = _get_sampler(sampling_type)

    def sample(
        self, batch_embs: np.ndarray, batch_qids: np.ndarray, negative_cnts: int
    ) -> np.ndarray:
        neighbors = self.searcher.find(
            batch_embs, max(negative_cnts + len(batch_embs), 100)
        )
        wanted_neighbors_mask = np.isin(self.qids[neighbors], batch_qids, invert=True)
        return self.sample_f(
            batch_qids, negative_cnts, neighbors, wanted_neighbors_mask
        )
