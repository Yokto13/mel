import logging
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.backends.cuda.matmul.allow_tf32 = True

from models.searchers.searcher import Searcher

_logger = logging.getLogger("models.searchers.brute_force_searcher")


class BruteForceSearcher(Searcher):
    def __init__(
        self, embs: np.ndarray, results: np.ndarray, run_build_from_init: bool = True
    ):
        if torch.cuda.is_available():
            _logger.info("Running on CUDA.")
            self.device: torch.device = torch.device("cuda")
        else:
            _logger.info("CUDA is not available.")
            self.device: torch.device = torch.device("cpu")
        super().__init__(embs, results, run_build_from_init)

    @torch.compile
    def find(self, batch: np.ndarray, num_neighbors: int) -> np.ndarray:
        with torch.no_grad():
            batch_torch: torch.Tensor = torch.from_numpy(batch).to(self.device)
            # embs after build are (dim, embs_count)
            dot_product: torch.Tensor = F.linear(batch_torch, self.embs)
            _, top_indices = dot_product.topk(num_neighbors)
        top_indices_np: np.ndarray = top_indices.cpu().numpy()
        return self.results[top_indices_np]

    def build(self) -> None:
        self.embs: torch.Tensor = torch.tensor(self.embs, device=self.device)


class _WrappedSearcher(nn.Module):
    def __init__(self, kb_embs, num_neighbors):
        super().__init__()
        self.kb_embs: torch.Tensor = nn.Parameter(kb_embs)
        self.num_neighbors: int = num_neighbors

    def forward(self, x):
        dot_product = F.linear(x, self.kb_embs)
        _, top_indices = dot_product.topk(self.num_neighbors)
        return top_indices


class DPBruteForceSearcher(Searcher):
    def __init__(
        self, embs: np.ndarray, results: np.ndarray, run_build_from_init: bool = True
    ):
        if torch.cuda.is_available():
            _logger.info("Running on CUDA.")
            self.device: torch.device = torch.device("cuda")
        else:
            _logger.info("CUDA is not available.")
            self.device: torch.device = torch.device("cpu")
        self.module_searcher: Optional[nn.DataParallel] = None
        self.required_num_neighbors: Optional[int] = None
        super().__init__(embs, results, run_build_from_init)

    def find(self, batch: np.ndarray, num_neighbors: int) -> np.ndarray:
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
            top_indices: torch.Tensor = self.module_searcher(
                torch.tensor(batch, device=self.device)
            )
        top_indices_np: np.ndarray = top_indices.cpu().numpy()
        return self.results[top_indices_np]

    def build(self):
        pass
