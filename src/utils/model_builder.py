from enum import StrEnum
import logging

from transformers import AutoModel
from torch import nn

from models.change_dim_wrapper import ChangeDimWrapper
from models.pooling_wrappers import (
    PoolerOutputWrapper,
    SentenceTransformerWrapper,
    CLSWrapper,
)

_logger = logging.getLogger("models.model_builder")


class ModelOutputType(StrEnum):
    PoolerOutput = "pooler_output"
    SENTENCE_TRANSFORMER = "sentence_transformer"
    CLS = "cls"


class ModelBuilder:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self._output_type = None
        self._dim = None
        self._model = None

    @property
    def output_type(self):
        return self._output_type

    @property
    def dim(self):
        return self._dim

    def set_output_type(self, output_type: ModelOutputType):
        _logger.info(f"Setting output type to {output_type}")
        self._output_type = output_type

    def set_dim(self, dim: int):
        _logger.info(f"Setting dim to {dim}")
        self._dim = dim

    def build(self) -> nn.Module:
        _logger.info("Building model")
        self._load_model()
        self._wrap_with_output_layer()
        self._enforce_output_dim()
        return self._model

    def _load_model(self):
        self._model = AutoModel.from_pretrained(self.model_path)

    def _wrap_with_output_layer(self):
        if self.output_type == ModelOutputType.PoolerOutput:
            self._model = PoolerOutputWrapper(self._model)
        elif self.output_type == ModelOutputType.CLS:
            self._model = CLSWrapper(self._model)
        elif self.output_type == ModelOutputType.SENTENCE_TRANSFORMER:
            self._model = SentenceTransformerWrapper(self._model)
        else:
            raise ValueError(f"Invalid output type: {self.output_type}")

    def _enforce_output_dim(self):
        if self.dim is None:
            return
        self._model = ChangeDimWrapper(self._model, self.dim)
