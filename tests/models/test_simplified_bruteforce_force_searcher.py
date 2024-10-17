import numpy as np
import pytest
import torch

from models.searchers import SimplifiedBruteForceSearcher


def test_search_present():
    embs = np.array(
        [
            [0.9, 0.9, 0.9],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    searcher = SimplifiedBruteForceSearcher(embs, np.arange(4))

    for i, e in enumerate(embs):
        res = searcher.find(np.array([e]), 2)
        assert res[0][0] == i
        assert res[0][1] != i
        assert len(res[0]) == 2


def test_search_missing():
    embs = np.array(
        [
            [1.0, 1.0, 1.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    searcher = SimplifiedBruteForceSearcher(embs, np.arange(4))

    res = searcher.find(np.array([[1.0, 0.0, 1.0]]), 2)
    assert res[0][0] == 0
