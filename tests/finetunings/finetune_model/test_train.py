from pathlib import PosixPath
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from finetunings.finetune_model.data import (
    _load_epoch_npz,
    LightWeightDataset,
    LinksAndDescriptionsTogetherDataset,
)
from finetunings.finetune_model.monitoring import batch_recall, get_wandb_logs
from finetunings.finetune_model.train import forward_to_embeddings
from finetunings.finetune_model.train_ddp import construct_labels
from transformers import AutoTokenizer
from utils.model_factory import ModelFactory
from utils.running_averages import RunningAverages


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

    recall = batch_recall(outputs, target, k=1)
    assert recall == 1.0  # Both correct answers are the top-1 prediction


def test_batch_recall_k_2():
    """Test recall at k=2 with mixed results."""
    outputs = torch.tensor([[0.1, 0.2, 0.7], [0.8, 0.1, 0.3]])
    target = torch.tensor([[0, 1, 0], [0, 0, 1]])

    recall = batch_recall(outputs, target, k=2)
    assert recall == 1.0


def test_batch_recall_k_too_large():
    """Test recall when k is larger than the number of outputs."""
    outputs = torch.tensor([[0.1, 0.2, 0.7], [0.8, 0.1, 0.3]])
    target = torch.tensor([[0, 1, 0], [0, 0, 1]])

    recall = batch_recall(outputs, target, k=4)
    assert recall == 0.0  # k is too large, so recall should be 0.0


def test_batch_recall_no_correct_predictions():
    """Test recall when no correct predictions are in the top k."""
    outputs = torch.tensor([[0.7, 0.2, 0.1], [0.6, 0.3, 0.1]])
    target = torch.tensor([[0, 0, 1], [0, 0, 1]])

    recall = batch_recall(outputs, target, k=2)
    assert recall == 0.0  # None of the correct answers are in the top-2


def test_batch_recall_large_k():
    """Test recall when k is very large, such as 100."""
    outputs = torch.tensor(np.arange(int(10**6)).reshape(1000, 1000))

    target = torch.zeros_like(outputs).scatter_(
        1, torch.zeros((10, 1), dtype=torch.long), 1
    )

    recall = batch_recall(outputs, target, k=100)

    assert recall == 0.0


def mock_load_epoch_npz(dataset_dir, epoch):
    links = np.array([[101, 102], [201, 202]])  # Example link tokens
    descriptions = np.array(
        [[301, 302, 303, 304], [401, 402, 403, 404]]  # Example description tokens
    )
    Y = np.array([1, 0])  # Example labels
    return links, descriptions, Y


def test_get_wandb_logs():
    # Arrange
    loss_item = 0.5
    r_at_1 = 0.7
    r_at_10 = 0.9

    # Create a mock RunningAverages object with expected properties
    mock_running_averages = MagicMock(spec=RunningAverages)
    mock_running_averages.loss = 0.4
    mock_running_averages.recall_1 = 0.6
    mock_running_averages.recall_10 = 0.8
    mock_running_averages.loss_big = 0.3
    mock_running_averages.recall_1_big = 0.5
    mock_running_averages.recall_10_big = 0.7

    # Act
    logs = get_wandb_logs(loss_item, r_at_1, r_at_10, mock_running_averages)

    # Assert
    expected_logs = {
        "loss": loss_item,
        "r_at_1": r_at_1,
        "r_at_10": r_at_10,
        "running_loss": mock_running_averages.loss,
        "running_r_at_1": mock_running_averages.recall_1,
        "running_r_at_10": mock_running_averages.recall_10,
        "running_loss_big": mock_running_averages.loss_big,
        "running_r_at_1_big": mock_running_averages.recall_1_big,
        "running_r_at_10_big": mock_running_averages.recall_10_big,
    }

    assert logs == expected_logs


class TestLinksAndDescriptionsTogetherDataset:
    @pytest.fixture
    def mock_data(self):
        links = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15]])
        descriptions = np.array(
            [
                [101, 102, 103, 104, 105],
                [106, 107, 108, 109, 110],
                [111, 112, 113, 114, 115],
            ]
        )
        Y = np.array([0, 1, 0])
        return links, descriptions, Y

    @pytest.fixture
    def mock_dataset(self, mock_data, tmp_path):
        links, descriptions, Y = mock_data

        # Create a mock NPZ file
        np.savez(tmp_path / "epoch_1.npz", X=links, lines=descriptions, Y=Y)

        with patch(
            "finetunings.finetune_model.data._load_epoch_npz",
            return_value=(links, descriptions, Y),
        ):
            dataset = LinksAndDescriptionsTogetherDataset(tmp_path, 1)
        return dataset

    def test_len(self, mock_dataset):
        assert len(mock_dataset) == 3

    def test_getitem(self, mock_dataset):
        item, label = mock_dataset[1]
        expected_item = np.concatenate(([6, 7, 8, 9, 10], [106, 107, 108, 109, 110]))
        assert np.array_equal(item, expected_item)
        assert label == 1

    def test_links_cnt(self, mock_dataset):
        assert mock_dataset.links_cnt == 5

    def test_descriptions_cnt(self, mock_dataset):
        assert mock_dataset.descriptions_cnt == 5


@pytest.mark.slow
class TestForwardToEmbeddings:
    @pytest.fixture(scope="module")
    def model_and_tokenizer(self):
        # Load the model and tokenizer
        model = ModelFactory.auto_load_from_file(
            "hf-internal-testing/tiny-random-BertModel"
        )
        tokenizer = AutoTokenizer.from_pretrained(
            "hf-internal-testing/tiny-random-BertModel"
        )
        return model, tokenizer

    @pytest.fixture(scope="module")
    def sample_texts(self):
        return [
            "This is a sample sentence.",
            "Another example of input text.",
            "Short text.",
        ]

    def test_forward_to_embeddings(self, model_and_tokenizer, sample_texts):
        model, tokenizer = model_and_tokenizer

        inputs = tokenizer(
            sample_texts, padding=True, truncation=True, return_tensors="pt"
        )
        input_ids = inputs["input_ids"]

        result = forward_to_embeddings(input_ids, model)

        assert result.shape == (len(sample_texts), model.output_dim)

        torch.testing.assert_allclose(
            torch.norm(result, p=2, dim=1),
            torch.ones(len(sample_texts)),
            atol=1e-4,
            rtol=1e-4,
        )


class TestLightWeightDataset:
    @pytest.fixture
    def mock_data(self):
        links = np.array(
            [[1, 2, 3, 4, 5, 0], [6, 7, 8, 9, 10, 0], [11, 12, 13, 14, 15, 0]]
        )
        descriptions = np.array(
            [
                [101, 102, 103, 104, 105, 0],
                [106, 107, 108, 109, 110, 0],
                [111, 112, 113, 114, 115, 0],
            ]
        )
        return links, descriptions

    def create_mock_dataset(self, mock_data, tmp_path, rank, world_size):
        links, descriptions = mock_data
        np.savez(tmp_path / "epoch_1.npz", X=links, lines=descriptions)
        return LightWeightDataset(tmp_path, 1, rank, world_size)

    @pytest.fixture
    def mock_dataset(self, mock_data, tmp_path):
        return self.create_mock_dataset(mock_data, tmp_path, 0, 1)

    @pytest.fixture
    def mock_dataset_rank_0(self, mock_data, tmp_path):
        return self.create_mock_dataset(mock_data, tmp_path, 0, 3)

    @pytest.fixture
    def mock_dataset_rank_2(self, mock_data, tmp_path):
        return self.create_mock_dataset(mock_data, tmp_path, 2, 3)

    @pytest.fixture
    def mock_dataset_rank_1(self, mock_data, tmp_path):
        return self.create_mock_dataset(mock_data, tmp_path, 1, 3)

    def test_len(
        self,
        mock_dataset,
        mock_dataset_rank_0,
        mock_dataset_rank_1,
        mock_dataset_rank_2,
    ):
        assert len(mock_dataset) == 3
        assert len(mock_dataset_rank_0) == 3
        assert len(mock_dataset_rank_1) == 3
        assert len(mock_dataset_rank_2) == 3

    def test_getitem_basic(self, mock_dataset):
        item = mock_dataset[1]
        expected_item = np.concatenate(
            ([6, 7, 8, 9, 10, 0], [106, 107, 108, 109, 110, 0])
        )
        assert np.array_equal(item, expected_item)

    def test_getitem_rank_0(self, mock_dataset_rank_0):
        item = mock_dataset_rank_0[1]
        expected_item = np.array([6, 7, 8, 9])
        assert np.array_equal(item, expected_item)

        item = mock_dataset_rank_0[0]
        expected_item = np.array([1, 2, 3, 4])
        assert np.array_equal(item, expected_item)

    def test_getitem_rank_1(self, mock_dataset_rank_1):
        item = mock_dataset_rank_1[1]
        expected_item = np.array([10, 0, 106, 107])
        assert np.array_equal(item, expected_item)

    def test_getitem_rank_2(self, mock_dataset_rank_2):
        item = mock_dataset_rank_2[1]
        expected_item = np.array([108, 109, 110, 0])
        assert np.array_equal(item, expected_item)

    def test_links_cnt(
        self,
        mock_dataset,
        mock_dataset_rank_0,
        mock_dataset_rank_1,
        mock_dataset_rank_2,
    ):
        assert mock_dataset.links_cnt == 6
        assert mock_dataset_rank_0.links_cnt == 6
        assert mock_dataset_rank_1.links_cnt == 6
        assert mock_dataset_rank_2.links_cnt == 6

    def test_descriptions_cnt(
        self,
        mock_dataset,
        mock_dataset_rank_0,
        mock_dataset_rank_1,
        mock_dataset_rank_2,
    ):
        assert mock_dataset.descriptions_cnt == 6
        assert mock_dataset_rank_0.descriptions_cnt == 6
        assert mock_dataset_rank_1.descriptions_cnt == 6
        assert mock_dataset_rank_2.descriptions_cnt == 6


class TestDDPUtils:
    def test_construct_labels(self):
        dataset = MagicMock(spec=LightWeightDataset)
        dataset.links_cnt = 3
        dataset.descriptions_cnt = 9

        labels = construct_labels(dataset)

        expected_labels = np.array(
            [
                [1, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0, 0],
            ],
            dtype=np.float32,
        )

        assert np.array_equal(labels, expected_labels)
