import numpy as np
import pytest
import torch

from models.searchers.brute_force_searcher import (
    BruteForceSearcher,
    DPBruteForceSearcher,
)


def test_search_present():
    embs = np.array(
        [
            [0.9, 0.9, 0.9],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    searcher = BruteForceSearcher(embs, np.arange(4))

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
    searcher = BruteForceSearcher(embs, np.arange(4))

    res = searcher.find(np.array([[1.0, 0.0, 1.0]]), 2)
    assert res[0][0] == 0


def test_search_large():
    embs = np.random.random((10000, 128))
    embs = embs / np.linalg.norm(embs, ord=2, axis=1, keepdims=True)
    searcher = BruteForceSearcher(embs, np.arange(len(embs)))
    neg = 7

    for i in range(1000):
        batch = np.random.random((32, 128))
        batch = batch / np.linalg.norm(batch, ord=2, axis=1, keepdims=True)
        res = searcher.find(batch, neg)
        for j, emb in enumerate(batch):
            neighbor_embs = embs[res[j]]
            dists = [emb @ ne for ne in neighbor_embs]
            dists_order = [dists[i] >= dists[i + 1] for i in range(len(dists) - 1)]
            assert all(dists_order)


class TestDPBruteForceSearcher:

    @pytest.fixture
    def small_embs(self):
        return np.array(
            [
                [0.9, 0.9, 0.9],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )

    @pytest.fixture
    def large_embs(self):
        embs = np.random.random((10000, 128))
        return embs / np.linalg.norm(embs, ord=2, axis=1, keepdims=True)

    def test_search_present(self, small_embs):
        searcher = DPBruteForceSearcher(small_embs, np.arange(4))
        for i, e in enumerate(small_embs):
            res = searcher.find(np.array([e]), 2)
            assert res[0][0] == i
            assert res[0][1] != i
            assert len(res[0]) == 2

    def test_search_missing(self):
        embs = np.array(
            [
                [1.0, 1.0, 1.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        searcher = DPBruteForceSearcher(embs, np.arange(4))
        res = searcher.find(np.array([[1.0, 0.0, 1.0]]), 2)
        assert res[0][0] == 0

    def test_search_large(self, large_embs):
        searcher = DPBruteForceSearcher(large_embs, np.arange(len(large_embs)))
        neg = 7
        for _ in range(10):  # Reduced iterations for faster testing
            batch = np.random.random((32, 128))
            batch = batch / np.linalg.norm(batch, ord=2, axis=1, keepdims=True)
            res = searcher.find(batch, neg)
            for j, emb in enumerate(batch):
                neighbor_embs = large_embs[res[j]]
                dists = [np.dot(emb, ne) for ne in neighbor_embs]
                dists_order = [dists[i] >= dists[i + 1] for i in range(len(dists) - 1)]
                assert all(dists_order)

    def test_device_selection(self, small_embs):
        searcher = DPBruteForceSearcher(small_embs, np.arange(len(small_embs)))
        expected_device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        assert searcher.device == expected_device

    def test_changing_num_neighbors(self, small_embs):
        searcher = DPBruteForceSearcher(small_embs, np.arange(len(small_embs)))
        searcher.find(np.random.random((1, 3)), 2)  # Initialize with 2 neighbors
        with pytest.raises(Exception):
            searcher.find(np.random.random((1, 3)), 3)  # Try to change to 3 neighbors

    def test_dataparallel_initialization(self, small_embs):
        searcher = DPBruteForceSearcher(small_embs, np.arange(len(small_embs)))
        searcher.find(
            np.random.random((1, 3)), 2
        )  # This should initialize module_searcher
        assert isinstance(searcher.module_searcher, torch.nn.DataParallel)
