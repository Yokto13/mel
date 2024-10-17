""" Primarly for testing purposes """

import numpy as np

from .searcher import Searcher


class SimplifiedBruteForceSearcher(Searcher):
    def __init__(
        self, embs: np.ndarray, results: np.ndarray, run_build_from_init: bool = True
    ):
        super().__init__(embs, results, run_build_from_init)

    def find(self, batch: np.ndarray, num_neighbors: int) -> np.ndarray:
        dot_product: np.ndarray = np.dot(batch, self.embs.T)
        top_indices = np.argsort(dot_product, axis=1)[:, -num_neighbors:][:, ::-1]
        return self.results[top_indices]

    def build(self) -> None:
        self.embs: np.ndarray = self.embs
