import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

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
        # dot_product = batch_torch @ self.embs  # (batch_size, embs_count)
        dot_product = F.linear(batch_torch, self.embs)
        _, top_indices = dot_product.topk(num_neighbors)
        top_indices = top_indices.cpu().numpy()
        return self.results[top_indices]

    def build(self):
        self.embs = torch.tensor(self.embs, device=self.device)


class _WrappedSearcher(nn.Module):
    def __init__(self, kb_embs, num_neighbors):
        super().__init__()
        self.kb_embs: torch.Tensor = nn.Parameter(kb_embs)
        self.num_neighbors: int = num_neighbors

    def forward(self, x):
        # dot_product = x @ self.kb_embs
        dot_product = F.linear(x, self.kb_embs)
        _, top_indices = dot_product.topk(self.num_neighbors)
        return top_indices


class DPBruteForceSearcher(Searcher):
    def __init__(
        self, embs: np.ndarray, results: np.ndarray, run_build_from_init: bool = True
    ):
        if torch.cuda.is_available():
            _logger.info("Running on CUDA.")
            self.device = torch.device("cuda")
        else:
            _logger.info("CUDA is not available.")
            self.device = torch.device("cpu")
        self.module_searcher = None
        super().__init__(embs, results, run_build_from_init)

    def find(self, batch, num_neighbors) -> np.ndarray:
        if self.module_searcher is None:
            self.module_searcher = nn.DataParallel(
                _WrappedSearcher(torch.from_numpy(self.embs), num_neighbors)
            )
            self.module_searcher.to(self.device)
            self.required_num_neighbors = num_neighbors
        if self.required_num_neighbors != num_neighbors:
            raise ValueError(
                f"num_neighbors was changed from {self.required_num_neighbors} to {num_neighbors} and this is not allowed in DPBruteForceSearcher"
            )
        with torch.no_grad():
            top_indices = self.module_searcher(torch.tensor(batch, device=self.device))
        top_indices = top_indices.cpu().numpy()
        return self.results[top_indices]

    def build(self):
        pass
