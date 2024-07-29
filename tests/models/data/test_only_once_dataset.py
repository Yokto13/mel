import numpy as np
from torch.utils.data import IterableDataset

# Import the classes you want to test
from models.data.only_once_dataset import (
    OnlyOnceDataset,
    _OnlyOnceTokens,
    _TokensHasher,
)


class MockIterableDataset(IterableDataset):
    def __init__(self, data):
        self.data = data

    def __iter__(self):
        return iter(self.data)


def test_only_once_dataset():
    mock_data = [
        (np.array([1, 2, 3]), "qid1"),
        (np.array([1, 2, 3]), "qid2"),
        (np.array([4, 5, 6]), "qid3"),
    ]
    mock_dataset = MockIterableDataset(mock_data)
    only_once_dataset = OnlyOnceDataset(mock_dataset)

    result = list(only_once_dataset)
    assert len(result) == 2
    assert np.array_equal(result[0][0], np.array([1, 2, 3]))
    assert result[0][1] == "qid1"
    assert np.array_equal(result[1][0], np.array([4, 5, 6]))
    assert result[1][1] == "qid3"


def test_only_once_tokens():
    tokens_db = _OnlyOnceTokens()

    # Test with initial tokens
    toks1 = np.array([1, 2, 3])
    result1 = tokens_db(toks1)
    assert np.array_equal(result1, toks1)

    # Test with repeated tokens
    result2 = tokens_db(toks1)
    assert result2 is None

    # Test with new tokens
    toks2 = np.array([4, 5, 6])
    result3 = tokens_db(toks2)
    assert np.array_equal(result3, toks2)


def test_tokens_hasher():
    hasher = _TokensHasher(3)

    # Test hash computation
    toks1 = np.array([1, 2, 3])
    hash1 = hasher(toks1)
    print(type(hash1))
    assert isinstance(hash1, np.int64)
    assert hash1 == (1 * 1 + 2 * 3 + 3 * 9)

    # Test hash consistency
    hash2 = hasher(toks1)
    assert hash1 == hash2

    # Test different tokens produce different hashes
    toks2 = np.array([4, 5, 6])
    hash3 = hasher(toks2)
    assert hash1 != hash3


def test_tokens_hasher_initialization():
    hasher = _TokensHasher(5)
    assert hasher.P == int(10**9 + 7)
    assert hasher.a == 3
    assert len(hasher.powers) == 5
    assert np.array_equal(hasher.powers, [1, 3, 9, 27, 81])


def test_only_once_tokens_edge_cases():
    tokens_db = _OnlyOnceTokens()

    # Test with empty array
    toks_empty = np.array([])
    result_empty = tokens_db(toks_empty)
    assert np.array_equal(result_empty, toks_empty)

    # Test with very large tokens
    toks_large = np.array([10**9, 10**9 + 1, 10**9 + 2])
    result_large = tokens_db(toks_large)
    assert np.array_equal(result_large, toks_large)


def test_only_once_tokens_long():
    tokens_db = _OnlyOnceTokens()

    toks_large = np.array(np.arange(10000))
    result_large = tokens_db(toks_large)
    assert np.array_equal(result_large, toks_large)


def test_only_once_dataset_empty():
    mock_data = []
    mock_dataset = MockIterableDataset(mock_data)
    only_once_dataset = OnlyOnceDataset(mock_dataset)

    result = list(only_once_dataset)
    assert len(result) == 0
