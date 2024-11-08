from enum import Enum
import logging
from unittest.mock import Mock

import numpy as np
import pytest
from models.negative_sampler import (
    _get_sampler,
    _sample_shuffling_numba,
    _sample_top_numba,
    NegativeSampler,
    NegativeSamplingType,
)
from models.searchers.scann_searcher import ScaNNSearcher


class MockSearcher(ScaNNSearcher):
    def __init__(self, embs, results):
        # super().__init__(embs, results)
        self.find = Mock()
        self.build = Mock()


@pytest.fixture(autouse=True)
def set_random_seed():
    np.random.seed(42)


@pytest.fixture
def sample_data():
    embs = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15]])
    qids = np.array([1, 2, 3, 4, 5])
    return embs, qids


@pytest.fixture
def negative_sampler(sample_data):
    embs, qids = sample_data
    return NegativeSampler(embs, qids, MockSearcher, NegativeSamplingType("top"))


def test_initialization(sample_data):
    embs, qids = sample_data
    sampler = NegativeSampler(embs, qids, MockSearcher, NegativeSamplingType("top"))

    assert np.array_equal(sampler.embs, embs)
    assert np.array_equal(sampler.qids, qids)
    assert isinstance(sampler.searcher, MockSearcher)


def test_initialization_with_mismatched_lengths():
    embs = np.array([[1, 2, 3], [4, 5, 6]])
    qids = np.array([1, 2, 3])

    with pytest.raises(AssertionError):
        NegativeSampler(embs, qids, MockSearcher, NegativeSamplingType("top"))


def test_sample_basic(negative_sampler):
    batch_embs = np.array([[1, 1, 1], [2, 2, 2]])
    batch_qids = np.array([1, 2])
    negative_cnts = 2

    negative_sampler.searcher.find.return_value = np.array([[1, 2, 3, 4], [1, 4, 0, 3]])

    result = negative_sampler.sample(batch_embs, batch_qids, negative_cnts)

    assert result.shape == (2, 2)
    for ans, expected in zip(result, [[1, 2, 3, 4], [2, 4, 0, 3]]):
        print(ans)
        for x in ans:
            assert x in set(expected)


@pytest.mark.slow
def test_sample_with_large_batch():
    negative_sampler = NegativeSampler(
        np.random.rand(10000, 128),
        np.random.default_rng().choice(20000, size=10000),
        MockSearcher,
        NegativeSamplingType("top"),
    )
    batch_embs = np.random.rand(1024, 128)
    batch_qids = np.random.randint(1, 1000, size=1024)
    negative_cnts = 7

    mock_neighbors = np.random.randint(0, 10000, size=(1024, 8 + 1024))
    negative_sampler.searcher.find.return_value = mock_neighbors

    result = negative_sampler.sample(batch_embs, batch_qids, negative_cnts)

    assert result.shape == (1024, 7)


def test_sample_edge_case_all_same_qid(negative_sampler):
    batch_embs = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]])
    batch_qids = np.array([1, 1, 1])
    negative_cnts = 2

    negative_sampler.searcher.find.return_value = np.array(
        [[0, 1, 2, 3, 4], [1, 0, 3, 4, 2], [2, 0, 3, 4, 1]]
    )

    result = negative_sampler.sample(batch_embs, batch_qids, negative_cnts)

    assert result.shape == (3, 2)
    for x in result:
        assert 0 not in x


def test_sample_randomness(negative_sampler):
    negative_sampler = NegativeSampler(
        np.random.rand(2000, 8),
        np.random.default_rng().choice(20000, size=2000),
        MockSearcher,
        NegativeSamplingType("shuffle"),
    )
    batch_embs = np.random.rand(1024, 8)
    batch_qids = np.random.randint(1, 1000, size=1024)
    negative_cnts = 7

    mock_neighbors = np.random.randint(0, 2000, size=(1024, 8 + 1024))
    negative_sampler.searcher.find.return_value = mock_neighbors

    result1 = negative_sampler.sample(batch_embs, batch_qids, negative_cnts)
    result2 = negative_sampler.sample(batch_embs, batch_qids, negative_cnts)

    assert not np.array_equal(result1, result2)


def test_negative_sampling_type():
    assert isinstance(NegativeSamplingType.Shuffling, Enum)
    assert isinstance(NegativeSamplingType.MostSimilar, Enum)
    assert NegativeSamplingType.Shuffling.value == "shuffle"
    assert NegativeSamplingType.MostSimilar.value == "top"


def test_get_sampler_shuffling():
    sampler = _get_sampler(NegativeSamplingType.Shuffling)
    assert sampler == _sample_shuffling_numba


def test_get_sampler_most_similar():
    sampler = _get_sampler(NegativeSamplingType.MostSimilar)
    assert sampler == _sample_top_numba


def test_get_sampler_invalid_type():
    with pytest.raises(AttributeError):
        _get_sampler("invalid_type")


@pytest.mark.parametrize("randomly_sampled", [10, 100])
@pytest.mark.parametrize(
    "sampling_type, qids_distribution, expected_warning",
    [
        (NegativeSamplingType.MostSimilarDistribution, None, True),
        (NegativeSamplingType.ShufflingDistribution, None, True),
        (NegativeSamplingType.MostSimilarDistribution, np.ones(5) / 5, False),
        (NegativeSamplingType.ShufflingDistribution, np.ones(5) / 5, False),
    ],
)
def test_negative_sampler_validation_warning(
    sampling_type: NegativeSamplingType,
    qids_distribution: np.ndarray,
    randomly_sampled: int,
    expected_warning: bool,
    caplog,
):
    embs = np.random.rand(5, 3)
    qids = np.arange(5)

    with caplog.at_level(logging.WARNING):
        NegativeSampler(
            embs,
            qids,
            ScaNNSearcher,
            sampling_type,
            qids_distribution,
            randomly_sampled,
        )

    if expected_warning:
        assert "qids_distribution is None" in caplog.text
    else:
        assert "qids_distribution is None" not in caplog.text


@pytest.mark.parametrize("randomly_sampled", [0.1, None])
@pytest.mark.parametrize(
    "sampling_type, qids_distribution",
    [
        (NegativeSamplingType.MostSimilarDistribution, np.ones(5) / 5),
        (NegativeSamplingType.ShufflingDistribution, np.ones(5) / 5),
    ],
)
def test_negative_sampler_validation_assert_error(
    sampling_type: NegativeSamplingType,
    qids_distribution: np.ndarray,
    randomly_sampled: int,
):
    embs = np.random.rand(5, 3)
    qids = np.arange(5)

    with pytest.raises(AssertionError):
        NegativeSampler(
            embs,
            qids,
            ScaNNSearcher,
            sampling_type,
            qids_distribution,
            randomly_sampled,
        )


@pytest.mark.parametrize("randomly_sampled", [None, 10])
@pytest.mark.parametrize("qids_distribution", [None, np.ones(5) / 5])
@pytest.mark.parametrize(
    "sampling_type",
    [
        NegativeSamplingType.MostSimilar,
        NegativeSamplingType.Shuffling,
    ],
)
def test_negative_sampler_validation_no_warning_or_error(
    sampling_type: NegativeSamplingType,
    qids_distribution: np.ndarray | None,
    randomly_sampled: int | None,
    caplog,
):
    embs = np.random.rand(5, 3)
    qids = np.arange(5)

    with caplog.at_level(logging.WARNING):
        NegativeSampler(
            embs,
            qids,
            ScaNNSearcher,
            sampling_type,
            qids_distribution,
            randomly_sampled,
        )

    assert len(caplog.records) == 0
