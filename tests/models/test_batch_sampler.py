import numpy as np
import pytest
from models.batch_sampler import BatchSampler
from models.negative_sampler import NegativeSamplingType

from models.searchers import SimplifiedBruteForceSearcher
from pytest import fixture


@fixture
def data_size():
    return 500


@fixture
def random_embs(data_size):
    embs = np.random.random((data_size, 16))
    embs = embs / np.linalg.norm(embs, ord=2, axis=1, keepdims=True)
    return embs


@fixture
def qids(data_size):
    return np.arange(data_size)


@fixture
def top_batch_sampler(random_embs, qids):
    return BatchSampler(
        random_embs,
        qids,
        SimplifiedBruteForceSearcher,
        NegativeSamplingType.MostSimilar,
    )


@fixture
def shuffle_batch_sampler(random_embs, qids):
    return BatchSampler(
        random_embs, qids, SimplifiedBruteForceSearcher, NegativeSamplingType.Shuffling
    )


@pytest.mark.parametrize(
    "batch_sampler_fixture", ["top_batch_sampler", "shuffle_batch_sampler"]
)
def test_positive_and_negatives_differ(batch_sampler_fixture, request, data_size):
    batch_sampler = request.getfixturevalue(batch_sampler_fixture)

    for _ in range(10):
        batch_embs = np.random.random((8, 16))
        batch_qids = np.random.randint(0, data_size, 8)
        pos, neg = batch_sampler.sample(batch_embs, batch_qids, 7)
        for p, n in zip(pos, neg):
            assert isinstance(p, np.int_)
            assert isinstance(n, np.ndarray)
            assert p not in n


def test_sampling_type_differences(top_batch_sampler, shuffle_batch_sampler, data_size):
    batch_embs = np.random.random((128, 16))
    batch_qids = np.random.randint(0, data_size, 32)

    top_pos, top_neg = top_batch_sampler.sample(batch_embs, batch_qids, 7)
    shuffle_pos, shuffle_neg = shuffle_batch_sampler.sample(batch_embs, batch_qids, 7)

    assert np.array_equal(
        top_pos, shuffle_pos
    ), "Positive samples should be the same for both samplers"
    assert not np.array_equal(
        top_neg, shuffle_neg
    ), "Negative samples should differ between samplers"
