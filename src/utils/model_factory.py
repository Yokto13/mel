import logging

import torch
from typing import Any

from transformers import BertModel

from src.models.smaller_dim import SmallerDim

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
    def load_bert_with_reduced_dim(
        cls, file_path: str, target_dim: int
    ) -> torch.nn.Module:
        bert_model = cls.load_bert_from_file(file_path)
        return SmallerDim(bert_model, target_dim)

    @classmethod
    def load_bert_with_reduced_dim_and_state_dict(
        cls, file_path: str, state_dict_path: str, target_dim: int
    ) -> torch.nn.Module:
        model = cls.load_bert_with_reduced_dim(file_path, target_dim)
        return cls._add_state_dict_to_model(state_dict_path, model)

    @classmethod
    def auto_load_from_file(
        cls,
        file_path: str,
        state_dict_path: str | None = None,
        target_dim: int | None = None,
    ) -> torch.nn.Module:
        if target_dim is None:
            return cls._load_without_target_dim(file_path, state_dict_path)
        else:
            return cls._load_with_target_dim(file_path, state_dict_path, target_dim)

    @classmethod
    def _load_without_target_dim(
        cls, file_path: str, state_dict_path: str | None
    ) -> torch.nn.Module:
        if state_dict_path is None or state_dict_path == "None": # TODO fix this
            return cls.load_bert_from_file(file_path)
        else:
            return cls.load_bert_from_file_and_state_dict(file_path, state_dict_path)

    @classmethod
    def _load_with_target_dim(
        cls, file_path: str, state_dict_path: str | None, target_dim: int
    ) -> torch.nn.Module:
        if state_dict_path is None or state_dict_path == "None":
            return cls.load_bert_with_reduced_dim(file_path, target_dim)
        else:
            return cls.load_bert_with_reduced_dim_and_state_dict(
                file_path, state_dict_path, target_dim
            )

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
