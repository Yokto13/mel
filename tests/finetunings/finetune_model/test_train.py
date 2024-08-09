from pathlib import Path, PosixPath
from unittest.mock import patch
import numpy as np
import pytest
import torch

from finetunings.finetune_model.train import (
    _load_epoch_npz,
    _batch_recall,
    _embeddig_gen,
    _get_links_and_descriptions_from_halves,
    _SplitToTwoDataset,
)


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


def check_embedding_gen_calls_embedding(cnt):
    with patch(
        "finetunings.finetune_model.train._forward_to_embeddings"
    ) as mocked_method:
        data = [None for i in range(cnt)]
        model = None
        g = _embeddig_gen(data, model)
        assert mocked_method.call_count == 0
        list(g)
        assert mocked_method.call_count == cnt


def test_embedding_gen_calls_embedding_once():
    check_embedding_gen_calls_embedding(1)


def test_embedding_gen_calls_embedding_a_lot():
    check_embedding_gen_calls_embedding(100)


def test_get_links_and_descriptions_from_halves_correct_behavior():
    first_half = torch.tensor([1, 2, 3, 4])
    second_half = torch.tensor([5, 6, 7, 8])
    links_cnt = 2

    links_embedded, descs_embedded = _get_links_and_descriptions_from_halves(
        first_half, second_half, links_cnt
    )

    assert torch.equal(links_embedded, torch.tensor([1, 2]))
    assert torch.equal(descs_embedded, torch.tensor([3, 4, 5, 6, 7, 8]))


def test_get_links_and_descriptions_from_halves_all_links():
    first_half = torch.tensor([1, 2, 3, 4])
    second_half = torch.tensor([5, 6, 7])
    links_cnt = 4

    links_embedded, descs_embedded = _get_links_and_descriptions_from_halves(
        first_half, second_half, links_cnt
    )

    assert torch.equal(links_embedded, torch.tensor([1, 2, 3, 4]))
    assert torch.equal(descs_embedded, torch.tensor([5, 6, 7]))


def test_get_links_and_descriptions_from_halves_links_count_too_high():
    first_half = torch.tensor([1, 2, 3, 4])
    second_half = torch.tensor([5, 6, 7, 8])
    links_cnt = 5  # more than the length of first_half

    with pytest.raises(ValueError):
        _get_links_and_descriptions_from_halves(first_half, second_half, links_cnt)


def mock_load_epoch_npz(dataset_dir, epoch):
    links = np.array([[101, 102], [201, 202]])  # Example link tokens
    descriptions = np.array(
        [[301, 302, 303, 304], [401, 402, 403, 404]]  # Example description tokens
    )
    Y = np.array([1, 0])  # Example labels
    return links, descriptions, Y


# Test cases
@patch(
    "finetunings.finetune_model.train._load_epoch_npz", side_effect=mock_load_epoch_npz
)
def test_split_to_two_dataset(mock_load_npz):
    dataset_dir = Path("/fake/dir")  # This is irrelevant due to mocking
    epoch = 1
    dataset = _SplitToTwoDataset(dataset_dir, epoch)

    # Test dataset length
    assert len(dataset) == 2

    # Test first sample
    first_half, second_half, y = dataset[0]
    assert np.array_equal(first_half, np.array([101, 102, 301]))
    assert np.array_equal(second_half, np.array([302, 303, 304]))
    assert y == 1

    # Test second sample
    first_half, second_half, y = dataset[1]
    assert np.array_equal(first_half, np.array([201, 202, 401]))
    assert np.array_equal(second_half, np.array([402, 403, 404]))
    assert y == 0


@patch(
    "finetunings.finetune_model.train._load_epoch_npz", side_effect=mock_load_epoch_npz
)
def test_split_to_two_dataset_no_mid_split(mock_load_npz):
    # This case tests when the mid-point exactly divides the descriptions
    links = np.array([[101, 102]])
    descriptions = np.array([[301, 302]])
    Y = np.array([1])

    mock_load_npz.side_effect = lambda dataset_dir, epoch: (links, descriptions, Y)

    dataset_dir = Path("/fake/dir")
    epoch = 1
    dataset = _SplitToTwoDataset(dataset_dir, epoch)

    first_half, second_half, y = dataset[0]
    assert np.array_equal(first_half, np.array([101, 102]))
    assert np.array_equal(second_half, np.array([301, 302]))
    assert y == 1


@patch(
    "finetunings.finetune_model.train._load_epoch_npz", side_effect=mock_load_epoch_npz
)
def test_split_to_two_dataset_links_count(mock_load_npz):
    dataset_dir = Path("/fake/dir")
    epoch = 1
    dataset = _SplitToTwoDataset(dataset_dir, epoch)

    assert dataset.links_cnt == 2


@patch(
    "finetunings.finetune_model.train._load_epoch_npz", side_effect=mock_load_epoch_npz
)
def test_split_to_two_dataset_descriptions_count(mock_load_npz):
    dataset_dir = Path("/fake/dir")
    epoch = 1
    dataset = _SplitToTwoDataset(dataset_dir, epoch)

    assert dataset.descriptions_cnt == 4
