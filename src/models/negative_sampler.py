from enum import Enum
import logging

import numba as nb
import numpy as np

from models.searchers.searcher import Searcher

_logger = logging.getLogger(__name__)


class NegativeSamplingType(Enum):
    Shuffling = "shuffle"
    MostSimilar = "top"
    MostSimilarDistribution = "top_distribution"
    ShufflingDistribution = "shuffling_distribution"


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
    if sampler_type in (
        NegativeSamplingType.Shuffling,
        NegativeSamplingType.ShufflingDistribution,
    ):
        return _sample_shuffling_numba
    if sampler_type in (
        NegativeSamplingType.MostSimilar,
        NegativeSamplingType.MostSimilarDistribution,
    ):
        return _sample_top_numba
    raise AttributeError(f"No samplig method for {sampler_type}")


class NegativeSampler:
    def __init__(
        self,
        embs: np.ndarray,
        qids: np.ndarray,
        searcher_constructor: type[Searcher],
        sampling_type: NegativeSamplingType,
        qids_distribution: np.ndarray | None = None,
        randomly_sampled_cnt: int | None = None,
    ) -> None:
        assert len(embs) == len(qids)
        self.embs = embs
        self.qids = qids
        # self.set_arr = np.zeros(int(np.max(self.qids)) + 1, dtype=np.bool_)
        self.returned_indices = np.arange(len(embs))
        self.searcher = searcher_constructor(embs, self.returned_indices)
        self.sampling_type = sampling_type
        self.sample_f = _get_sampler(sampling_type)
        self.qids_distribution = qids_distribution
        self.randomly_sampled_cnt = randomly_sampled_cnt
        self._validate()

    def sample(
        self, batch_embs: np.ndarray, batch_qids: np.ndarray, negative_cnts: int
    ) -> np.ndarray:
        if self._should_sample_randomly():
            negative_cnts -= self.randomly_sampled_cnt
        neighbors = self.searcher.find(
            batch_embs, max(negative_cnts + len(batch_embs), 100)
        )
        # performance seems comparable with _get_neighbors_mask_set_arr
        # by the Occams razor _get_neighbors_mask_set is better.
        wanted_neighbors_mask = _get_neighbors_mask_set(
            batch_qids, self.qids[neighbors]
        )
        sampled = self.sample_f(
            batch_qids, negative_cnts, neighbors, wanted_neighbors_mask
        )
        if self._should_sample_randomly():
            randomly_sampled = self._sample_randomly(batch_qids, negative_cnts)
            sampled = np.concatenate([sampled, randomly_sampled], axis=1)
        return sampled

    def _should_sample_randomly(self):
        return self.sampling_type in (
            NegativeSamplingType.ShufflingDistribution,
            NegativeSamplingType.MostSimilarDistribution,
        )

    def _sample_randomly(self, batch_qids, negative_cnts):
        batch_size = len(batch_qids)
        batch_qid = batch_qids[0]
        batch_qids = set(batch_qids)
        result = np.empty(batch_size, negative_cnts)
        for i in range(batch_size):
            for j in range(negative_cnts):
                qid_to_add = batch_qid
                while qid_to_add in batch_qids:
                    qid_idx = np.random.choice(
                        self.returned_indices, size=1, p=self.qids_distribution
                    )[0]
                    qid_to_add = self.qids[qid_idx]
                result[i][j] = qid_idx
        return result

    def _validate(self):
        if self._should_sample_randomly():
            if self.qids_distribution is None:
                _logger.warning(
                    "qids_distribution is None, negative sampling will use uniform distribution."
                )
                self.qids_distribution = np.ones(len(self.qids)) / len(self.qids)
            assert isinstance(self.randomly_sampled_cnt, int)
