import numba
import numpy as np

from models.searchers.searcher import Searcher


@numba.njit
def _sample_numba(batch_qids, negative_cnts, neighbors, neighbors_mask):
    res = np.empty((len(batch_qids), negative_cnts), dtype=np.int32)

    # print(neighbors)
    # print(batch_qids)
    for i in range(len(batch_qids)):
        res[i] = neighbors[i][neighbors_mask[i]][:negative_cnts]
        # print("i", i)
        # print("mask", neighbors_mask[i])
        # negative_neighbors = neighbors[i][neighbors_mask[i]]
        # print("negative_neighbors", negative_neighbors)
        # min_to_sample = min(negative_neighbors.shape[0], 12 * negative_cnts)
        # print("min_to_sample", min_to_sample)
        # p = np.random.permutation(min_to_sample)
        # most_similar = negative_neighbors[:min_to_sample]
        # print("most_similar", most_similar)
        # most_similar_permuted = most_similar[p]
        # print("most_similar_permuted", most_similar_permuted)
        # res[i] = most_similar_permuted[:negative_cnts]
        # print("=================")

    return res


class NegativeSampler:
    def __init__(
        self, embs: np.ndarray, qids: np.ndarray, searcher_constructor: type[Searcher]
    ) -> None:
        assert len(embs) == len(qids)
        self.embs = embs
        self.qids = qids
        self.searcher = searcher_constructor(embs, np.arange(len(embs)))

    def sample(
        self, batch_embs: np.ndarray, batch_qids: np.ndarray, negative_cnts: int
    ) -> np.ndarray:
        neighbors = self.searcher.find(
            batch_embs, max(negative_cnts + len(batch_embs), 150)
        )
        neighbors_mask = np.isin(self.qids[neighbors], batch_qids, invert=True)
        return _sample_numba(batch_qids, negative_cnts, neighbors, neighbors_mask)
