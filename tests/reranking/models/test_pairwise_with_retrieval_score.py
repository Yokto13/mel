from __future__ import annotations

import math
from typing import Sequence

import pytest
import torch
from torch import nn
from transformers.tokenization_utils_base import BatchEncoding

from reranking.models import PairwiseMLPRerankerWithRetrievalScore
from utils.embeddings import create_attention_mask

MENTION_TEXTS = ["dummy mention positive", "dummy mention negative"]
ENTITY_TEXTS = ["dummy entity positive", "dummy entity negative"]


class DummyEmbeddingModel(nn.Module):
    output_dim = 2

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        masked = input_ids.float() * attention_mask
        summed = masked.sum(dim=1, keepdim=True)
        return torch.cat([summed, 2 * summed], dim=1)


class DummyTokenizer:
    _TOKEN_MAP = {
        MENTION_TEXTS[0]: [1, 1, 0, 0],
        MENTION_TEXTS[1]: [1, 0, 0, 0],
        ENTITY_TEXTS[0]: [3, 0, 0, 0],
        ENTITY_TEXTS[1]: [0, 0, 0, 0],
    }

    def __call__(
        self,
        texts: str | Sequence[str],
        *,
        padding: bool = True,
        truncation: bool = True,
        return_tensors: str = "pt",
    ) -> BatchEncoding:
        del padding, truncation, return_tensors
        items = [texts] if isinstance(texts, str) else list(texts)
        encodings = [self._TOKEN_MAP.get(text, [1, 0, 0, 0]) for text in items]
        tensor = torch.tensor(encodings, dtype=torch.long)
        attention_mask = (tensor != 0).long()
        return BatchEncoding({"input_ids": tensor, "attention_mask": attention_mask})


def _configure_classifier(model: PairwiseMLPRerankerWithRetrievalScore) -> None:
    hidden_layer = model.classifier[0]
    output_layer = model.classifier[-1]

    with torch.no_grad():
        hidden_layer.weight.copy_(torch.tensor([[0.1, 0.2, 0.3, 0.4]], dtype=torch.float))
        hidden_layer.bias.copy_(torch.tensor([0.5], dtype=torch.float))
        output_layer.weight.copy_(torch.tensor([[1.0]], dtype=torch.float))
        output_layer.bias.zero_()


def _prepare_labels(size: int, *, positive_index: int = 0) -> torch.Tensor:
    labels = torch.zeros(size, dtype=torch.float)
    labels[positive_index] = 1.0
    return labels


@pytest.fixture()
def dummy_model(monkeypatch) -> PairwiseMLPRerankerWithRetrievalScore:
    dummy_embedding_model = DummyEmbeddingModel()

    def _mock_model_loader(*_args, **_kwargs):
        return dummy_embedding_model

    monkeypatch.setattr(
        "reranking.models.pairwise_mlp.ModelFactory.auto_load_from_file", _mock_model_loader
    )
    monkeypatch.setattr(
        "reranking.models.pairwise_mlp.AutoTokenizer.from_pretrained",
        lambda *_args, **_kwargs: DummyTokenizer(),
    )

    model = PairwiseMLPRerankerWithRetrievalScore(
        "dummy-model",
        mlp_hidden_dim=1,
        dropout=0.0,
    )
    _configure_classifier(model)
    return model


def test_score_not_failing(dummy_model: PairwiseMLPRerankerWithRetrievalScore) -> None:
    score = dummy_model.score(MENTION_TEXTS[0], ENTITY_TEXTS[0])
    assert isinstance(score, float)
    assert score >= 0.0
    assert score <= 1.0
