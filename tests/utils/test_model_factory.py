from unittest.mock import patch

from transformers import AutoTokenizer, AutoModel
import torch

from models.pooling_wrappers import PoolerOutputWrapper
from utils.model_factory import ModelFactory


def test_auto_load_from_file_default():
    model = ModelFactory.auto_load_from_file("setu4993/LEALLA-base")

    assert model is not None
    assert isinstance(model, PoolerOutputWrapper)


def test_auto_load_from_file_smaller_dim():
    model = ModelFactory.auto_load_from_file("setu4993/LEALLA-base", target_dim=10)

    assert model is not None
    assert model.mapping.out_features == 10


@patch("torch.load")
def test_auto_load_from_file_state_dict_and_target_dim(mock_torch_load, tmp_path):
    model = ModelFactory.auto_load_from_file("setu4993/LEALLA-base", target_dim=10)
    state_dict_path = f"{tmp_path}/final.pth"

    mock_state_dict = model.state_dict()
    mock_torch_load.return_value = mock_state_dict

    model = ModelFactory.auto_load_from_file(
        "setu4993/LEALLA-base", state_dict_path=state_dict_path, target_dim=10
    )

    assert model is not None
    mock_torch_load.assert_called_once_with(state_dict_path, map_location="cpu")


@patch("torch.load")
def test_auto_load_from_file_state_dict_and_not_target_dim(mock_torch_load, tmp_path):
    model = ModelFactory.auto_load_from_file("setu4993/LEALLA-base")
    state_dict_path = f"{tmp_path}/final.pth"

    mock_state_dict = model.state_dict()
    mock_torch_load.return_value = mock_state_dict

    model = ModelFactory.auto_load_from_file(
        "setu4993/LEALLA-base", state_dict_path=state_dict_path
    )

    assert model is not None
    mock_torch_load.assert_called_once_with(state_dict_path, map_location="cpu")


def test_auto_load_with_passthrough():
    model = ModelFactory.auto_load_from_file(
        "setu4993/LEALLA-base", output_type="sentence_transformer", target_dim=12
    )

    tokenizer = AutoTokenizer.from_pretrained("setu4993/LEALLA-base")
    tokenized = tokenizer("Hello world", return_tensors="pt")

    output = model(tokenized.input_ids, tokenized.attention_mask)
    assert output.shape == (1, 12)


def test_auto_load_from_file_state_dict_old(tmp_path):
    original_model = AutoModel.from_pretrained("setu4993/LEALLA-base")
    state_dict_path = f"{tmp_path}/final.pth"

    torch.save(original_model.state_dict(), state_dict_path)

    model = ModelFactory.auto_load_from_file(
        "setu4993/LEALLA-base", state_dict_path=state_dict_path
    )

    assert model is not None
    assert isinstance(model, PoolerOutputWrapper)
    base_model = model.model
    assert base_model.config.hidden_size == original_model.config.hidden_size
