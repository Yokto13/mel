""" Wrapper around any searcher we might use. """

from abc import ABC, abstractmethod
import logging

import numpy as np

_logger = logging.getLogger("models.negative_index")


class Searcher(ABC):
    def __init__(
        self, embs: np.ndarray, results: np.ndarray, run_build_from_init: bool
    ):
        assert len(embs) == len(results)
        self.embs = embs
        self.results = results
        if run_build_from_init:
            self.build()

    @abstractmethod
    def find(self, batch, num_neighbors) -> np.ndarray:
        pass

    @abstractmethod
    def build(self):
        pass
