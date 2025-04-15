from unittest.mock import MagicMock

import numpy as np
import pytest
from pathlib import PosixPath
import torch

from finetunings.finetune_model.data import _load_epoch_npz
from finetunings.finetune_model.monitoring import (
    batch_recall,
    get_gradient_norm,
    _get_wandb_logs,
    process_metrics,
)
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


class TestBatchRecall:
    def test_k_1(self):
        """Test recall at k=1 with simple cases."""
        outputs = torch.tensor([[0.1, 0.2, 0.9], [0.8, 0.1, 0.3]])
        target = torch.tensor([[0, 0, 1], [1, 0, 0]])

        recall = batch_recall(outputs, target, k=1)
        assert recall == 1.0  # Both correct answers are the top-1 prediction

    def test_k_2(self):
        """Test recall at k=2 with mixed results."""
        outputs = torch.tensor([[0.1, 0.2, 0.7], [0.8, 0.1, 0.3]])
        target = torch.tensor([[0, 1, 0], [0, 0, 1]])

        recall = batch_recall(outputs, target, k=2)
        assert recall == 1.0

    def test_k_too_large(self):
        """Test recall when k is larger than the number of outputs."""
        outputs = torch.tensor([[0.1, 0.2, 0.7], [0.8, 0.1, 0.3]])
        target = torch.tensor([[0, 1, 0], [0, 0, 1]])

        recall = batch_recall(outputs, target, k=4)
        assert recall == 0.0  # k is too large, so recall should be 0.0

    def test_no_correct_predictions(self):
        """Test recall when no correct predictions are in the top k."""
        outputs = torch.tensor([[0.7, 0.2, 0.1], [0.6, 0.3, 0.1]])
        target = torch.tensor([[0, 0, 1], [0, 0, 1]])

        recall = batch_recall(outputs, target, k=2)
        assert recall == 0.0  # None of the correct answers are in the top-2

    def test_large_k(self):
        """Test recall when k is very large, such as 100."""
        outputs = torch.tensor(np.arange(int(10**6)).reshape(1000, 1000))

        target = torch.zeros_like(outputs).scatter_(
            1, torch.zeros((10, 1), dtype=torch.long), 1
        )

        recall = batch_recall(outputs, target, k=100)

        assert recall == 0.0


class TestWandbLogs:
    @pytest.fixture
    def mock_running_averages(self):
        mock_running_averages = MagicMock(spec=RunningAverages)
        mock_running_averages.loss = 0.4
        mock_running_averages.recall_1 = 0.6
        mock_running_averages.recall_10 = 0.8
        mock_running_averages.loss_big = 0.3
        mock_running_averages.recall_1_big = 0.5
        mock_running_averages.recall_10_big = 0.7
        return mock_running_averages

    @pytest.fixture
    def loss_item(self):
        return 0.5

    @pytest.fixture
    def r_at_1(self):
        return 0.7

    @pytest.fixture
    def r_at_10(self):
        return 0.9

    def test_no_kwargs(self, loss_item, r_at_1, r_at_10, mock_running_averages):
        logs = _get_wandb_logs(loss_item, r_at_1, r_at_10, mock_running_averages)

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

    def test_with_kwargs(self, loss_item, r_at_1, r_at_10, mock_running_averages):
        extra_kwargs = {"extra_metric": 0.95, "another_metric": 0.85}
        logs = _get_wandb_logs(
            loss_item, r_at_1, r_at_10, mock_running_averages, **extra_kwargs
        )

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
            **extra_kwargs,
        }

        assert logs == expected_logs


class TestGradientNorm:
    @pytest.fixture
    def model(self):
        return torch.nn.Linear(10, 5)

    def test_nonzero(self, model):
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        input_data = torch.randn(2, 10)
        target = torch.randn(2, 5)

        output = model(input_data)
        loss = torch.nn.functional.mse_loss(output, target)

        optimizer.zero_grad()
        loss.backward()

        # Repeat to make sure that get_gradient_norm does not mess up the model.
        for _ in range(3):
            norm = get_gradient_norm(model)
            assert isinstance(norm, float)
            assert norm > 0

    def test_zero(self, model):
        norm = get_gradient_norm(model)
        assert norm is None


class TestProcessMetrics:
    @pytest.fixture
    def outputs(self):
        return torch.tensor([[0.1, 0.2, 0.7], [0.8, 0.1, 0.3]])

    @pytest.fixture
    def labels(self):
        return torch.tensor([[0, 1, 0], [1, 0, 0]])

    @pytest.fixture
    def loss_item(self):
        return 0.5

    @pytest.fixture
    def running_averages(self):
        mock_running_averages = MagicMock(spec=RunningAverages)
        mock_running_averages.loss = 0.4
        mock_running_averages.recall_1 = 0.6
        mock_running_averages.recall_10 = 0.8
        mock_running_averages.loss_big = 0.3
        mock_running_averages.recall_1_big = 0.5
        mock_running_averages.recall_10_big = 0.7
        return mock_running_averages

    @pytest.mark.parametrize(
        "additional_metrics",
        [
            {},
            {"extra_metric": 0.95},
            {"extra_metric": 0.95, "another_metric": 0.85},
        ],
    )
    def test_process_metrics_with_wandb_log(
        self, outputs, labels, loss_item, running_averages, mocker, additional_metrics
    ):
        mock_wandb_log = mocker.patch("wandb.log")

        process_metrics(
            outputs=outputs,
            labels=labels,
            loss_item=loss_item,
            running_averages=running_averages,
            additional_metrics=additional_metrics,
        )

        mock_wandb_log.assert_called_once()
        logged_data = mock_wandb_log.call_args[0][0]

        assert "loss" in logged_data
        assert "r_at_1" in logged_data
        assert "r_at_10" in logged_data
        assert logged_data["loss"] == loss_item
        assert logged_data["running_loss"] == running_averages.loss
        assert logged_data["running_r_at_1"] == running_averages.recall_1
        assert logged_data["running_r_at_10"] == running_averages.recall_10

        for key, value in additional_metrics.items():
            assert key in logged_data
            assert logged_data[key] == value
