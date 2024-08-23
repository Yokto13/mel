import logging

import numpy as np
import torch

from models.searchers.searcher import Searcher

_logger = logging.getLogger("models.searchers.brute_force_searcher")


class BruteForceSearcher(Searcher):
    def __init__(
        self, embs: np.ndarray, results: np.ndarray, run_build_from_init: bool = True
    ):
        if torch.cuda.is_available():
            _logger.info("Running on CUDA.")
            self.device = torch.device("cuda")
        else:
            _logger.info("CUDA is not available.")
            self.device = torch.device("cpu")
        super().__init__(embs, results, run_build_from_init)

    def find(self, batch, num_neighbors) -> np.ndarray:
        batch_torch = torch.tensor(batch, device=self.device)
        # batch is (batch_size, dim)
        # embs after build are (dim, embs_count)
        dot_product = batch_torch @ self.embs  # (batch_size, embs_count)
        _, top_indices = dot_product.topk(num_neighbors)
        top_indices = top_indices.cpu().numpy()
        return self.results[top_indices]

    def build(self):
        self.embs = torch.tensor(self.embs, device=self.device).T
