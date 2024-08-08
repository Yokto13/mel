import pytest
import numpy as np
from unittest.mock import Mock
from models.searcher import ScaNNSearcher
from models.negative_sampler import NegativeSampler


class MockSearcher(ScaNNSearcher):
    def __init__(self, embs, results):
        # super().__init__(embs, results)
        self.find = Mock()
        self.build = Mock()


@pytest.fixture
def sample_data():
    embs = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15]])
    qids = np.array([1, 2, 3, 4, 5])
    return embs, qids


@pytest.fixture
def negative_sampler(sample_data):
    embs, qids = sample_data
    return NegativeSampler(embs, qids, MockSearcher)


def test_initialization(sample_data):
    embs, qids = sample_data
    sampler = NegativeSampler(embs, qids, MockSearcher)

    assert np.array_equal(sampler.embs, embs)
    assert np.array_equal(sampler.qids, qids)
    assert isinstance(sampler.searcher, MockSearcher)


def test_initialization_with_mismatched_lengths():
    embs = np.array([[1, 2, 3], [4, 5, 6]])
    qids = np.array([1, 2, 3])

    with pytest.raises(AssertionError):
        NegativeSampler(embs, qids, MockSearcher)


def test_sample_basic(negative_sampler):
    batch_embs = np.array([[1, 1, 1], [2, 2, 2]])
    batch_qids = np.array([1, 2])
    negative_cnts = 2

    negative_sampler.searcher.find.return_value = np.array([[1, 2, 3, 4], [1, 4, 0, 3]])

    result = negative_sampler.sample(batch_embs, batch_qids, negative_cnts)

    assert result.shape == (2, 2)
    assert np.array_equal(result, np.array([[2, 3], [4, 3]]))


def test_sample_with_all_different_qids(negative_sampler):
    batch_embs = np.array([[1, 1, 1], [2, 2, 2]])
    batch_qids = np.array([4, 5])  # Different QIDs from those in Searcher
    negative_cnts = 2

    negative_sampler.searcher.find.return_value = np.array([[0, 1, 2, 3], [3, 1, 2, 4]])

    result = negative_sampler.sample(batch_embs, batch_qids, negative_cnts)

    assert result.shape == (2, 2)
    assert np.array_equal(result, np.array([[0, 1], [1, 2]]))


def test_sample_with_large_batch():
    negative_sampler = NegativeSampler(
        np.random.rand(10000, 128),
        np.random.default_rng().choice(20000, size=10000),
        MockSearcher,
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
    assert np.array_equal(result, np.array([[1, 2], [1, 3], [2, 3]]))
