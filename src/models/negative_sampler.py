import numba
import numpy as np

from models.searchers.searcher import Searcher


@numba.njit
def _sample_numba(batch_qids, negative_cnts, neighbors_mask, neighbors):
    res = np.empty((len(batch_qids), negative_cnts), dtype=np.int32)

    for i in range(len(batch_qids)):
        res[i] = neighbors[i][neighbors_mask[i]][:negative_cnts]

    return res


class NegativeSampler:
    def __init__(
        self, embs: np.ndarray, qids: np.ndarray, searcher_constructor: type[Searcher]
    ) -> None:
        assert len(embs) == len(qids)
        self.embs = embs
        self.qids = qids
        self.searcher = searcher_constructor(embs, np.arange(len(embs)))
        self.negative_mask = None

    def sample(
        self, batch_embs: np.ndarray, batch_qids: np.ndarray, negative_cnts: int
    ) -> np.ndarray:
        neighbors = self.searcher.find(batch_embs, negative_cnts + len(batch_embs))
        neighbors_mask = np.isin(self.qids[neighbors], batch_qids, invert=True)
        return _sample_numba(batch_qids, negative_cnts, neighbors_mask, neighbors)
