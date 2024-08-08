from pathlib import PosixPath
import numpy as np
import pytest
import torch

from finetunings.finetune_model.train import _load_epoch_npz, _batch_recall


@pytest.fixture
def sample_npz(tmp_path):
    """Fixture to create a sample .npz file."""
    file_path = tmp_path / "epoch_1.npz"

    # Sample data to be stored in the .npz file
    X_data = np.array([1, 2, 3])
    lines_data = np.array([1, 1, 1, 1])
    Y_data = np.array([4, 5, 6])

    # Save the data to the .npz file
    np.savez(file_path, X=X_data, lines=lines_data, Y=Y_data)

    return file_path


def test_load__epoch_npz(sample_npz: PosixPath):
    """Test that load_npz_file correctly loads and returns data."""
    X, lines, Y = _load_epoch_npz(sample_npz.parent, 1)

    assert np.array_equal(X, np.array([1, 2, 3]))
    assert np.array_equal(lines, np.array([1, 1, 1, 1]))
    assert np.array_equal(Y, np.array([4, 5, 6]))


def test_batch_recall_k_1():
    """Test recall at k=1 with simple cases."""
    outputs = torch.tensor([[0.1, 0.2, 0.9], [0.8, 0.1, 0.3]])
    target = torch.tensor([[0, 0, 1], [1, 0, 0]])

    recall = _batch_recall(outputs, target, k=1)
    assert recall == 1.0  # Both correct answers are the top-1 prediction


def test_batch_recall_k_2():
    """Test recall at k=2 with mixed results."""
    outputs = torch.tensor([[0.1, 0.2, 0.7], [0.8, 0.1, 0.3]])
    target = torch.tensor([[0, 1, 0], [0, 0, 1]])

    recall = _batch_recall(outputs, target, k=2)
    assert recall == 1.0


def test_batch_recall_k_too_large():
    """Test recall when k is larger than the number of outputs."""
    outputs = torch.tensor([[0.1, 0.2, 0.7], [0.8, 0.1, 0.3]])
    target = torch.tensor([[0, 1, 0], [0, 0, 1]])

    recall = _batch_recall(outputs, target, k=4)
    assert recall == 0.0  # k is too large, so recall should be 0.0


def test_batch_recall_no_correct_predictions():
    """Test recall when no correct predictions are in the top k."""
    outputs = torch.tensor([[0.7, 0.2, 0.1], [0.6, 0.3, 0.1]])
    target = torch.tensor([[0, 0, 1], [0, 0, 1]])

    recall = _batch_recall(outputs, target, k=2)
    assert recall == 0.0  # None of the correct answers are in the top-2


def test_batch_recall_large_k():
    """Test recall when k is very large, such as 100."""
    outputs = torch.tensor(np.arange(int(10**6)).reshape(1000, 1000))

    target = torch.zeros_like(outputs).scatter_(
        1, torch.zeros((10, 1), dtype=torch.long), 1
    )

    recall = _batch_recall(outputs, target, k=100)

    assert recall == 0.0
