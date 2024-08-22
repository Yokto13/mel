import logging

import torch
from typing import Any

import pytest
from transformers import BertModel

_logger = logging.getLogger("utils.model_factory")


class ModelFactory:
    @classmethod
    def load_bert_from_file(cls, file_path: str) -> torch.nn.Module:
        return BertModel.from_pretrained(file_path)

    @classmethod
    def load_bert_from_file_and_state_dict(
        cls, file_path: str, state_dict_path: str
    ) -> torch.nn.Module:
        model = cls.load_bert_from_file(file_path)
        return cls._add_state_dict_to_model(state_dict_path, model)

    @classmethod
    def _add_state_dict_to_model(
        cls, state_dict_path: str, model: torch.nn.Module
    ) -> torch.nn.Module:
        d = torch.load(state_dict_path)
        problematic_keys = model.load_state_dict(d)
        if len(problematic_keys.missing_keys) or len(problematic_keys.unexpected_keys):
            _logger.warning(
                f"The following keys are missing in the model {problematic_keys}"
            )
        return model


@pytest.fixture(scope="module")
def original_model():
    return ModelFactory.load_bert_from_file("setu4993/LEALLA-base")


def test_model_state_dict_modification(tmp_path, original_model):
    original_state_dict = original_model.state_dict()
    original_state_dict_path = tmp_path / "original_state_dict.pth"
    torch.save(original_state_dict, original_state_dict_path)

    with torch.no_grad():
        for param in original_model.parameters():
            param.add_(1.0)  # Add 1.0 to all parameters to modify them

    modified_state_dict_path = tmp_path / "modified_state_dict.pth"
    torch.save(original_model.state_dict(), modified_state_dict_path)

    reloaded_model = ModelFactory.load_bert_from_file_and_state_dict(
        "setu4993/LEALLA-base", str(modified_state_dict_path)
    )

    original_state_dict_reloaded = torch.load(original_state_dict_path)

    for original_param, reloaded_param in zip(
        original_state_dict_reloaded.values(), reloaded_model.state_dict().values()
    ):
        assert not torch.equal(
            original_param, reloaded_param
        ), "Parameters should differ after modification"
