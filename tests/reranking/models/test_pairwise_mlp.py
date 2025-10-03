from __future__ import annotations

import math
from typing import Sequence

import pytest
import torch
from torch import nn
from transformers.tokenization_utils_base import BatchEncoding

from reranking.models import PairwiseMLPReranker
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


def _configure_classifier(model: PairwiseMLPReranker) -> None:
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


def _run_common_checks(
    model: PairwiseMLPReranker,
    mentions: Sequence[str],
    entities: Sequence[str],
    labels: torch.Tensor,
):
    tokenizer = model.tokenizer
    mention_batch = tokenizer(mentions, padding=True, truncation=True, return_tensors="pt").to(
        model.device
    )
    mention_batch["attention_mask"] = create_attention_mask(mention_batch["input_ids"])
    entity_batch = tokenizer(entities, padding=True, truncation=True, return_tensors="pt").to(
        model.device
    )
    entity_batch["attention_mask"] = create_attention_mask(entity_batch["input_ids"])

    encode_out = model._encode(mention_batch["input_ids"])

    loss = model.train_step(
        {
            "mention_tokens": dict(mention_batch),
            "entity_tokens": dict(entity_batch),
            "labels": labels.to(model.device),
        }
    )

    with torch.no_grad():
        manual_mention_embeddings = model.base_model(
            input_ids=mention_batch["input_ids"],
            attention_mask=mention_batch.get("attention_mask"),
        )
        manual_entity_embeddings = model.base_model(
            input_ids=entity_batch["input_ids"],
            attention_mask=entity_batch.get("attention_mask"),
        )
        manual_logits = model.classifier(
            torch.cat([manual_mention_embeddings, manual_entity_embeddings], dim=-1)
        ).squeeze(-1)
        expected_loss = model.loss_fn(manual_logits, labels.to(model.device))

    score = model.score(mentions[0], entities[0])

    return {
        "encode": encode_out,
        "loss": loss.detach(),
        "score": score,
        "manual_mention_embeddings": manual_mention_embeddings,
        "manual_entity_embeddings": manual_entity_embeddings,
        "manual_logits": manual_logits,
        "expected_loss": expected_loss,
    }


@pytest.fixture()
def dummy_model(monkeypatch) -> PairwiseMLPReranker:
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

    model = PairwiseMLPReranker(
        "dummy-model",
        mlp_hidden_dim=1,
        dropout=0.0,
    )
    _configure_classifier(model)
    return model


def test_pairwise_mlp_with_dummy_backbone(dummy_model: PairwiseMLPReranker) -> None:
    labels = _prepare_labels(len(MENTION_TEXTS))
    results = _run_common_checks(dummy_model, MENTION_TEXTS, ENTITY_TEXTS, labels)

    assert torch.allclose(
        results["encode"],
        results["manual_mention_embeddings"],
    )

    assert torch.allclose(results["loss"], results["expected_loss"])

    positive_logit = results["manual_logits"][0].item()
    assert math.isclose(results["score"], torch.sigmoid(torch.tensor(positive_logit)).item())


@pytest.mark.slow
def test_pairwise_mlp_with_lealla_backbone() -> None:
    model = PairwiseMLPReranker(
        "setu4993/LEALLA-base",
        mlp_hidden_dim=1,
        dropout=0.0,
    )

    labels = _prepare_labels(len(MENTION_TEXTS))
    results = _run_common_checks(model, MENTION_TEXTS, ENTITY_TEXTS, labels)

    assert torch.allclose(
        results["encode"],
        results["manual_mention_embeddings"],
    )

    assert torch.allclose(results["loss"], results["expected_loss"], atol=1e-6)

    positive_probability = torch.sigmoid(results["manual_logits"][0]).item()
    assert 0.0 <= results["score"] <= 1.0
    assert math.isclose(results["score"], positive_probability, rel_tol=1e-5)
