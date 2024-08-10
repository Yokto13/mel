import torch
from typing import Any

from transformers import BertModel


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
        model.load_state_dict(d)
        return model
