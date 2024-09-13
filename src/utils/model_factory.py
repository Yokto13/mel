import logging

import torch
from typing import Any

from transformers import BertModel

from models.change_dim_wrapper import ChangeDimWrapper
from models.pooling_wrappers import PoolingWrapper
from utils.model_builder import OutputType, ModelBuilder

_logger = logging.getLogger("utils.model_factory")


class ModelFactory:
    # @classmethod
    # def load_bert_from_file(cls, file_path: str) -> torch.nn.Module:
    #     return BertModel.from_pretrained(file_path)

    # @classmethod
    # def load_bert_from_file_and_state_dict(
    #     cls, file_path: str, state_dict_path: str
    # ) -> torch.nn.Module:
    #     model = cls.load_bert_from_file(file_path)
    #     return cls._add_state_dict_to_model(state_dict_path, model)

    # @classmethod
    # def load_bert_with_reduced_dim(
    #     cls, file_path: str, target_dim: int
    # ) -> torch.nn.Module:
    #     bert_model = cls.load_bert_from_file(file_path)
    #     return ChangeDimWrapper(bert_model, target_dim)

    # @classmethod
    # def load_bert_with_reduced_dim_and_state_dict(
    #     cls, file_path: str, state_dict_path: str, target_dim: int
    # ) -> torch.nn.Module:
    #     model = cls.load_bert_with_reduced_dim(file_path, target_dim)
    #     return cls._add_state_dict_to_model(state_dict_path, model)

    @classmethod
    def auto_load_from_file(
        cls,
        file_path: str,
        state_dict_path: str | None = None,
        target_dim: int | None = None,
        output_type: OutputType | None = None,
    ) -> torch.nn.Module:
        builder = ModelBuilder(file_path)
        if output_type is None:
            output_type = OutputType.CLS  # the original/old default
        builder.set_output_type(output_type)
        if target_dim is not None:
            builder.set_dim(target_dim)
        model = builder.build()
        if state_dict_path is not None:
            model = cls._add_state_dict_to_model(state_dict_path, model)
        return model

    @classmethod
    def _add_state_dict_to_model(
        cls, state_dict_path: str, model: torch.nn.Module
    ) -> torch.nn.Module:
        d = torch.load(state_dict_path)
        try:
            model.load_state_dict(d)
        except RuntimeError as e:
            _logger.warning(
                f"Failed to load state dict: {e}, trying to load it the old way."
            )
            if isinstance(model, PoolingWrapper):
                model.model.load_state_dict(d)
                _logger.warning("Loaded state dict into base model.")
            else:
                raise e
        return model
