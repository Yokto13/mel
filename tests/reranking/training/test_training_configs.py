from unittest.mock import MagicMock

import pytest

from reranking.training.training_configs import TrainingConfig


def test_get_output_path_creates_directory(tmp_path):
    output_root = tmp_path / "outputs"
    config = TrainingConfig(
        config_name="test_config",
        model=MagicMock(),
        dataset=MagicMock(),
        optimizer=MagicMock(),
        save_each=100,
        batch_size=1,
        output_dir=str(output_root),
        validate_each=50,
    )

    path = config.get_output_path(step=5)

    expected_dir = output_root / "test_config"
    assert expected_dir.exists() and expected_dir.is_dir()
    assert path == f"{expected_dir}/5.pth"
