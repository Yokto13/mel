"""Currently BROKEN."""

import faiss
import numpy as np

from models.searchers.searcher import Searcher


class FaissSearcher(Searcher):
    def __init__(self, embs: np.ndarray, results: np.ndarray):
        super().__init__(embs, results, True)

    def find(self, batch, num_neighbors) -> np.ndarray:
        print(self.index.search(batch, num_neighbors))
        return self.results[self.index.search(batch, num_neighbors)]

    def build(self):
        self.build_index()

    def build_index(
        self,
        num_leaves=1000,
    ):
        dim = self.embs.shape[-1]
        quantizer = faiss.IndexFlatIP(dim)
        self.index = faiss.IndexIVFFlat(quantizer, dim, num_leaves)
        assert not self.index.is_trained
        self.index.train(self.embs)
        assert self.index.is_trained
