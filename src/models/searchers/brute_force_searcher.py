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

    def find(self, batch: np.ndarray, num_neighbors: int, mask=None) -> np.ndarray:
        # @torch.compile
        def _find(batch: np.ndarray) -> np.ndarray:
            batch_torch: torch.Tensor = torch.from_numpy(batch).to(self.device)
            # embs after build are (dim, embs_count)
            dot_product: torch.Tensor = F.linear(batch_torch, self.embs)
            _, top_indices = dot_product.topk(num_neighbors)
            return top_indices

        with torch.inference_mode():
            top_indices: torch.Tensor = _find(batch)

        top_indices_np: np.ndarray = top_indices.cpu().numpy()
        return self.results[top_indices_np]

    def build(self) -> None:
        self.embs: torch.Tensor = torch.from_numpy(self.embs).to(self.device)


class _WrappedSearcher(nn.Module):
    def __init__(self, kb_embs, num_neighbors):
        super().__init__()
        self.kb_embs: torch.Tensor = nn.Parameter(kb_embs)
        self.num_neighbors: int = num_neighbors

    # @torch.compile
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

    @torch.compile
    def find(self, batch: np.ndarray, num_neighbors: int, mask=None) -> np.ndarray:
        """
        Finds the nearest neighbors for a given batch of input data.
        CAREFUL: This is an optimized version that comes with potential pitfalls to get better performance.
        Read Notes for details!

        Args:
            batch (np.ndarray): A batch of input data for which neighbors are to be found.
            num_neighbors (int): The number of nearest neighbors to retrieve.
        Returns:
            np.ndarray: An array containing the results corresponding to the nearest neighbors.
        Raises:
            TypeError: If `module_searcher` if an unexpected attribute access occurs when using module_searcher.
        Notes:
            - It is not possible to change num_neighbors after the first call to find.
              If you need to do that, you need to reinitialize this object. If you call the find with different
              num_neighbors, it will not raise an error and will fail silently.
            - The first call to find will be slow, because the module_searcher will be initialized and torch.compile is called.
        """
        # with torch.inference_mode(), torch.autocast(
        #     device_type=self.device.type, dtype=torch.float16
        # ):
        with torch.no_grad():
            # A try except trick to avoid the overhead of checking if the module_searcher is None
            # on every call to find.
            # This is a bit of a hack, but it should make things faster as we are suggesting that the module_searcher is initialized.
            try:
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    top_indices: torch.Tensor = self.module_searcher(
                        torch.from_numpy(batch).to(self.device)
                    )
            except TypeError as e:
                if self.module_searcher is not None:
                    raise e
                self.module_searcher = nn.DataParallel(
                    _WrappedSearcher(torch.from_numpy(self.embs), num_neighbors)
                )
                self.module_searcher.to(self.device)
                self.required_num_neighbors = num_neighbors
                top_indices: torch.Tensor = self.module_searcher(
                    torch.from_numpy(batch).to(self.device)
                )

        top_indices_np: np.ndarray = top_indices.cpu().numpy()
        return self.results[top_indices_np]

    def build(self):
        pass
