from unittest.mock import patch, MagicMock

import pytest
import torch

from utils.model_factory import ModelFactory


@patch("transformers.BertModel.from_pretrained")
def test_load_bert_from_file(mock_from_pretrained):
    # Arrange
    fake_model = MagicMock(spec=torch.nn.Module)
    mock_from_pretrained.return_value = fake_model

    # Act
    model = ModelFactory.load_bert_from_file("/fake/path/to/model.pth")

    # Assert
    mock_from_pretrained.assert_called_once_with("/fake/path/to/model.pth")
    assert model is fake_model


@patch("transformers.BertModel.from_pretrained")
def test_load_bert_from_file_and_state_dict_with_invalid_path(mock_from_pretrained):
    # Arrange
    mock_from_pretrained.side_effect = FileNotFoundError

    # Act & Assert
    with pytest.raises(FileNotFoundError):
        ModelFactory.load_bert_from_file_and_state_dict(
            "/invalid/path/to/model.pth", {}
        )

    mock_from_pretrained.assert_called_once_with("/invalid/path/to/model.pth")


@patch("transformers.BertModel.from_pretrained")
def test_load_bert_from_file_with_invalid_path(mock_from_pretrained):
    # Arrange
    mock_from_pretrained.side_effect = FileNotFoundError

    # Act & Assert
    with pytest.raises(FileNotFoundError):
        ModelFactory.load_bert_from_file("/invalid/path/to/model.pth")

    mock_from_pretrained.assert_called_once_with("/invalid/path/to/model.pth")


def test_load_real_bert_from_hub():
    # Act
    model = ModelFactory.load_bert_from_file("setu4993/LEALLA-small")

    output_dim = model.pooler.dense.out_features
    assert output_dim == 128
