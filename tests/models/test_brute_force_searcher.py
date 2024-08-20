import numpy as np
import torch
from models.searchers.brute_force_searcher import BruteForceSearcher


def test_search_present():
    embs = np.array(
        [
            [1.0, 1.0, 1.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
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
