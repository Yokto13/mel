import faiss
import numpy as np

from models.searchers.searcher import Searcher


class FaissSearcher(Searcher):
    def __init__(self, embs: np.ndarray, results: np.ndarray):
        super().__init__(embs, results)

    def find(self, batch, num_neighbors) -> np.ndarray:
        return self.index.search(batch, num_neighbors)

    def build(self):
        self.build_index()

    def build_index(
        self,
        num_leaves=5000,
    ):
        dim = self.embs.shape[-1]
        quantizer = faiss.IndexFlatIP(dim)
        self.index = faiss.IndexIVFFlat(quantizer, dim, num_leaves)
        assert not self.index.is_trained
        self.index.train(self.embs)
        assert self.index.is_trained
