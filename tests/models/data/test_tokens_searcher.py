import pytest
import numpy as np
from models.data.tokens_searcher import (
    TokensSearcher,
)


@pytest.fixture
def sample_searcher():
    tokens = np.array([[1, 2, 4], [1, 3, 1], [2, 1, 1], [2, 1, 2], [1, 2, 3]])
    metadata = np.array(["B", "C", "D", "E", "A"])
    return TokensSearcher(tokens, metadata)


def test_initialization(sample_searcher):
    assert sample_searcher.tokens.shape == (5, 3)
    assert sample_searcher.metadata.shape == (5,)
    assert np.array_equal(
        sample_searcher.tokens,
        np.array([[1, 2, 3], [1, 2, 4], [1, 3, 1], [2, 1, 1], [2, 1, 2]]),
    )
    assert np.array_equal(sample_searcher.metadata, np.array(["A", "B", "C", "D", "E"]))


def test_find_existing_token(sample_searcher):
    assert sample_searcher.find(np.array([1, 2, 3])) == "A"
    assert sample_searcher.find(np.array([2, 1, 2])) == "E"


def test_find_nonexistent_token(sample_searcher):
    with pytest.raises(KeyError):
        sample_searcher.find(np.array([3, 3, 3]))


def test_single_token_searcher():
    single_searcher = TokensSearcher(np.array([[1, 2, 3]]), np.array(["X"]))
    assert single_searcher.find(np.array([1, 2, 3])) == "X"
    with pytest.raises(KeyError):
        single_searcher.find(np.array([1, 2, 4]), save=True)
    single_searcher.find(np.array([1, 2, 4]), save=False)


def test_large_dataset():
    n = 10000
    tokens = np.random.randint(0, 100, size=(n, 256))
    metadata = np.array([f"Item_{i}" for i in range(n)])
    large_searcher = TokensSearcher(tokens, metadata)

    # Test a random existing token
    random_index = np.random.randint(0, n)
    result = large_searcher.find(tokens[random_index])
    assert result == f"Item_{random_index}"

    # Test a likely non-existent token
    with pytest.raises(KeyError):
        large_searcher.find(np.array([101, 101, 101]), save=True)
