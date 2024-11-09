from collections import Counter
from enum import Enum
import json

import numba as nb
import numpy as np

from models.searchers.searcher import Searcher


class NegativeSamplingType(Enum):
    Shuffling = "shuffle"
    MostSimilar = "top"


# _rng = np.random.default_rng(seed=42)


@nb.njit(parallel=True)
def _sample_shuffling_numba(batch_qids, negative_cnts, neighbors, neighbors_mask):
    res = np.empty((len(batch_qids), negative_cnts), dtype=np.int32)
    for i in nb.prange(len(batch_qids)):
        ns = neighbors[i][neighbors_mask[i]]
        # _rng is not working with numba
        # _rng.shuffle(ns)
        np.random.shuffle(ns)
        res[i] = ns[:negative_cnts]
    return res


# currently too complicated without any real performance gain
# @nb.njit(parallel=True)
# def _sample_shuffling_numba(batch_qids, negative_cnts, neighbors, neighbors_mask):
#     res = np.empty((len(batch_qids), negative_cnts), dtype=np.int32)
#     max_neighbors = neighbors.shape[1]

#     for i in nb.prange(len(batch_qids)):
#         ns = np.empty(max_neighbors, dtype=np.int32)
#         valid_count = 0
#         for j in nb.prange(max_neighbors):
#             if neighbors_mask[i, j]:
#                 ns[valid_count] = neighbors[i, j]
#                 valid_count += 1

#         # Fisher-Yates shuffle implementation
#         for k in range(valid_count - 1, 0, -1):
#             j = np.random.randint(0, k + 1)
#             ns[k], ns[j] = ns[j], ns[k]

#         res[i] = ns[:negative_cnts]

#     return res


@nb.njit
def _sample_top_numba(batch_qids, negative_cnts, neighbors, neighbors_mask):
    res = np.empty((len(batch_qids), negative_cnts), dtype=np.int32)
    for i in range(len(batch_qids)):
        ns = neighbors[i][neighbors_mask[i]]
        res[i] = ns[:negative_cnts]
    return res


@nb.njit(parallel=True)
def _get_neighbors_mask_set_arr(batch_qids, neighbors_qids, set_arr):
    """
    set_arr is a boolean array with the same length as the number of unique qids in the dataset.
    It is used to mark the qids that are prohibited to sample.

    neighbors_qids is a 2D array of qids that are neighbors of the batch_qids.
    For each batch_qid it contains row of neighbor qids.

    Note: this is a linear numba implementation of the isin operation.
    The use of set_arr that is randomly accessed looks painful, and benchmarks shows
    that the size of it is crucial for the performance.
    For qids upto a few millions it should fit into L3 cache, for larger the performance
    drops significantly.
    """
    set_arr[batch_qids] = True  # these are prohibited to sample
    out = np.empty(neighbors_qids.shape, dtype=np.bool_)

    for i in nb.prange(neighbors_qids.shape[0]):
        for j in nb.prange(neighbors_qids.shape[1]):
            if set_arr[neighbors_qids[i][j]]:
                out[i][j] = False
            else:
                out[i][j] = True

    # reset the set_arr for the next batch
    set_arr[batch_qids] = False
    return out


@nb.njit(parallel=True)
def _get_neighbors_mask_set(batch_qids, neighbors_qids):
    prohibited_set = set(batch_qids)
    out = np.empty(neighbors_qids.shape, dtype=np.bool_)
    for i in nb.prange(neighbors_qids.shape[0]):
        for j in nb.prange(neighbors_qids.shape[1]):
            if neighbors_qids[i][j] in prohibited_set:
                out[i][j] = False
            else:
                out[i][j] = True

    return out


def _get_sampler(sampler_type: NegativeSamplingType) -> callable:
    if sampler_type == NegativeSamplingType.Shuffling:
        return _sample_shuffling_numba
    if sampler_type == NegativeSamplingType.MostSimilar:
        return _sample_top_numba
    raise AttributeError(f"No samplig method for {sampler_type}")


class NegativeSampler:
    def __init__(
        self,
        embs: np.ndarray,
        qids: np.ndarray,
        searcher_constructor: type[Searcher],
        sampling_type: NegativeSamplingType,
        verbose: bool = True,
    ) -> None:
        assert len(embs) == len(qids)
        self.embs = embs
        self.qids = qids
        # self.set_arr = np.zeros(int(np.max(self.qids)) + 1, dtype=np.bool_)
        self.searcher = searcher_constructor(embs, np.arange(len(embs)))
        print(sampling_type)
        self.sample_f = _get_sampler(sampling_type)
        self._mined_qids = Counter()
        self._mined_qids_total = 0
        self._verbose = verbose

    def sample(
        self, batch_embs: np.ndarray, batch_qids: np.ndarray, negative_cnts: int
    ) -> np.ndarray:
        neighbors = self.searcher.find(
            batch_embs, max(negative_cnts + len(batch_embs), 100)
        )
        # performance seems comparable with _get_neighbors_mask_set_arr
        # by the Occams razor _get_neighbors_mask_set is better.
        wanted_neighbors_mask = _get_neighbors_mask_set(
            batch_qids, self.qids[neighbors]
        )

        res = self.sample_f(batch_qids, negative_cnts, neighbors, wanted_neighbors_mask)

        if self._verbose:
            for i in range(len(res)):
                for idx in res[i]:
                    self._mined_qids[int(self.qids[idx])] += 1
            self._mined_qids_total += 1
            if self._mined_qids_total % 1000 == 0:
                print("Top 100 most common QIDs:")
                print(f"Total mined qids: {self._mined_qids_total}")
                for qid, count in self._mined_qids.most_common(100):
                    print(f"QID: {qid}, Count: {count}")
                print("\nBottom 20 least common QIDs:")
                for qid, count in sorted(self._mined_qids.items(), key=lambda x: x[1])[
                    :20
                ]:
                    print(f"QID: {qid}, Count: {count}")
                with open("mined_qids.json", "w") as f:
                    json.dump(
                        {"total": self._mined_qids_total, "counts": self._mined_qids},
                        f,
                        default=str,
                    )

        return res
