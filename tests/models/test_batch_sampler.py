import numpy as np
from pytest import fixture

from models.searchers.brute_force_searcher import BruteForceSearcher
from models.batch_sampler import BatchSampler


@fixture
def data_size():
    return 100000


@fixture
def random_embs(data_size):
    embs = np.random.random((data_size, 128))
    embs = embs / np.linalg.norm(embs, ord=2, axis=1, keepdims=True)
    return embs


@fixture
def qids(data_size):
    return np.arange(data_size)


@fixture
def random_batch_sampler(random_embs, qids):
    return BatchSampler(random_embs, qids, BruteForceSearcher)


def test_positive_and_negatives_differ(random_batch_sampler, data_size):
    for i in range(100):
        batch_embs = np.random.random((32, 128))
        batch_qids = np.random.randint(0, data_size, 32)
        pos, neg = random_batch_sampler.sample(batch_embs, batch_qids, 7)
        for p, n in zip(pos, neg):
            assert isinstance(p, np.int_)
            assert type(n) == np.ndarray
            assert p not in n
