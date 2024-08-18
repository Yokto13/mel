import numba
import numpy as np

from models.searchers.searcher import Searcher


@numba.njit
def _sample_numba(batch_qids, negative_cnts, neighbors, neighbors_mask):
    res = np.empty((len(batch_qids), negative_cnts), dtype=np.int32)

    for i in range(len(batch_qids)):
        negative_neighbors = neighbors[i][neighbors_mask[i]]
        min_to_sample = min(negative_neighbors.shape[0], 12 * negative_cnts)
        p = np.random.permutation(min_to_sample)
        most_similar = negative_neighbors[:min_to_sample]
        most_similar_permuted = most_similar[p]
        res[i] = most_similar_permuted[:negative_cnts]

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
