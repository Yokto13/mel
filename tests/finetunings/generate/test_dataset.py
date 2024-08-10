import numpy as np
import pytest
from pathlib import Path
from unittest.mock import patch

from finetunings.generate_epochs.datasets import Batcher

# Sample data for testing
sample_embs = np.random.rand(10000, 64)
sample_qids = np.arange(10000)
sample_tokens = np.random.randint(0, 100, (10000, 20))


def mock_load_embs_qids_tokens(dir_path):
    return sample_embs, sample_qids, sample_tokens


@patch(
    "finetunings.generate_epochs.datasets.load_embs_qids_tokens",
    side_effect=mock_load_embs_qids_tokens,
)
def test_shuffling(mock_load_fn):
    batch_size = 2
    known_qids = np.arange(10000)
    batcher = Batcher(Path("some/path"), known_qids, batch_size)

    initial_index = batcher._data_index.copy()
    batcher.shuffler()
    shuffled_index = batcher._data_index

    # Check that the indices are shuffled
    assert not np.array_equal(initial_index, shuffled_index)
    assert sorted(initial_index) == sorted(shuffled_index)


@patch(
    "finetunings.generate_epochs.datasets.load_embs_qids_tokens",
    side_effect=mock_load_embs_qids_tokens,
)
def test_get_batch(mock_load_fn):
    known_qids = np.array([1, 2, 3, 4, 5])
    batch_size = 2
    batcher = Batcher(Path("some/path"), known_qids, batch_size)

    batch1 = batcher.get_batch()
    assert len(batch1) == 3  # (embs, qids, tokens)
    assert batch1[0].shape == (batch_size, sample_embs.shape[1])
    assert batch1[1].shape == (batch_size,)
    assert batch1[2].shape == (batch_size, sample_tokens.shape[1])

    # Make sure batcher resets and shuffles when the end is reached
    batcher.get_batch()  # get the second batch
    initial_index = batcher._data_index.copy()
    batcher.get_batch()  # This should trigger a shuffle
    new_index = batcher._data_index

    # Check that a shuffle occurred
    assert not np.array_equal(initial_index, new_index)


@patch(
    "finetunings.generate_epochs.datasets.load_embs_qids_tokens",
    side_effect=mock_load_embs_qids_tokens,
)
def test_remove_when_qid_missing(mock_load_fn):
    batcher = Batcher(Path("some/path"), [], 2)
    known_qids = np.array([1, 3, 5, 7, 9])

    embs, qids, tokens = batcher._remove_when_qid_missing(
        (sample_embs, sample_qids, sample_tokens), known_qids
    )

    assert len(embs) == 5
    assert len(qids) == 5
    assert len(tokens) == 5

    assert np.array_equal(qids, known_qids)


@patch(
    "finetunings.generate_epochs.datasets.load_embs_qids_tokens",
    side_effect=mock_load_embs_qids_tokens,
)
def test_iter_basic(mock_load_fn):
    batch_size = 2
    known_qids = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    batcher = Batcher(Path("some/path"), known_qids, batch_size)

    seen_qids = set()

    for i, batch in enumerate(batcher):
        # Check the first batch
        assert len(batch) == 3  # Should return a tuple of (embs, qids, tokens)
        assert batch[0].shape == (batch_size, sample_embs.shape[1])  # embeddings
        assert batch[1].shape == (batch_size,)  # qids
        assert batch[2].shape == (batch_size, sample_tokens.shape[1])  # tokens
        seen_qids.add(batch[1][0])
        seen_qids.add(batch[1][1])
        if i == 30:
            break
    assert len(seen_qids) == 10
