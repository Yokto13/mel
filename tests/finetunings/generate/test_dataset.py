import random
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch

# Set random seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

from finetunings.generate_epochs.datasets import BatcherDataset

# Sample data for testing
sample_embs = np.random.rand(1000, 64)
sample_qids = np.random.randint(0, 100, (1000,))
sample_tokens = np.random.randint(0, 100, (1000, 20))


def mock_load_embs_qids_tokens(dir_path):
    return sample_embs, sample_qids, sample_tokens


@pytest.fixture
def num_files():
    return 10


@pytest.fixture
def dir_with_npz_files(tmpdir, num_files):
    dir_path = Path(tmpdir)
    for i in range(num_files):
        np.savez(
            dir_path / f"{i}.npz",
            embs=sample_embs,
            qids=sample_qids,
            tokens=sample_tokens,
        )
    return dir_path


@patch(
    "finetunings.generate_epochs.datasets.load_embs_qids_tokens",
    side_effect=mock_load_embs_qids_tokens,
)
def test_known_qids(mock_load_fn, dir_with_npz_files):
    batch_size = 2
    known_qids = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    batcher = BatcherDataset(dir_with_npz_files, known_qids, batch_size)

    seen_qids = set()

    for i, batch in enumerate(batcher):
        # Check the first batch
        assert len(batch) == 3  # Should return a tuple of (embs, qids, tokens)
        assert batch[0].shape == (batch_size, sample_embs.shape[1])  # embeddings
        assert batch[1].shape == (batch_size,)  # qids
        assert batch[2].shape == (batch_size, sample_tokens.shape[1])  # tokens
        seen_qids.add(batch[1][0])
        seen_qids.add(batch[1][1])
        if i == 1000:
            break
    assert len(seen_qids) == 10


@patch(
    "finetunings.generate_epochs.datasets.load_embs_qids_tokens",
    side_effect=mock_load_embs_qids_tokens,
)
def test_initial_shuffle(mock_load_fn, dir_with_npz_files):
    batch_size = 2
    known_qids = np.arange(100)
    batcher = BatcherDataset(dir_with_npz_files, known_qids, batch_size)

    first_batch = next(iter(batcher))

    batcher = BatcherDataset(dir_with_npz_files, known_qids, batch_size)
    second_batch = next(iter(batcher))

    assert not np.array_equal(
        first_batch[0], second_batch[0]
    ), "Initial shuffle did not occur"


@patch(
    "finetunings.generate_epochs.datasets.load_embs_qids_tokens",
    side_effect=mock_load_embs_qids_tokens,
)
def test_no_qid_repetition_in_batch(mock_load_fn, dir_with_npz_files):
    batch_size = 20
    known_qids = np.arange(1000)
    batcher = BatcherDataset(dir_with_npz_files, known_qids, batch_size)

    for i, batch in enumerate(batcher):
        qids_in_batch = batch[1]
        unique_qids = np.unique(qids_in_batch)

        assert len(qids_in_batch) == len(
            unique_qids
        ), "QIDs are repeating within a batch"
        assert (
            len(qids_in_batch) == batch_size
        ), "Batch size doesn't match expected size"

        if i == 10:
            break


@patch(
    "finetunings.generate_epochs.datasets.load_embs_qids_tokens",
    side_effect=mock_load_embs_qids_tokens,
)
def test_known_qids(mock_load_fn, dir_with_npz_files):
    batch_size = 2
    known_qids = np.array([1, 2])
    batcher = BatcherDataset(dir_with_npz_files, known_qids, batch_size)

    seen_qids = set()

    for i, batch in enumerate(batcher):
        seen_qids.add(batch[1][0])
        seen_qids.add(batch[1][1])
        if i == 100:
            break
    assert len(seen_qids) == len(known_qids)


@patch(
    "finetunings.generate_epochs.datasets.load_embs_qids_tokens",
    side_effect=mock_load_embs_qids_tokens,
)
@pytest.mark.parametrize("batch_size", [1, 2, 10, 16, 32, 64, 100])
def test_batch_sizes(mock_load_fn, dir_with_npz_files, batch_size):
    known_qids = np.arange(100)
    batcher = BatcherDataset(dir_with_npz_files, known_qids, batch_size)

    for i, batch in enumerate(batcher):
        for j in range(3):
            assert len(batch[j]) == batch_size
        if i == 10:
            break
