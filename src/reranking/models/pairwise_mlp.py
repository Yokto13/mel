from __future__ import annotations

from typing import Any, Dict, Mapping

import torch
from torch import nn
from transformers import AutoTokenizer

from reranking.models.base import BaseRerankingModel
from utils.embeddings import create_attention_mask
from utils.model_factory import ModelFactory, ModelOutputType


def _maybe_convert_output_type(output_type: ModelOutputType | str | None) -> ModelOutputType | None:
    if output_type is None or isinstance(output_type, ModelOutputType):
        return output_type
    return ModelOutputType(output_type)


def _infer_output_dim(model: nn.Module) -> int:
    if hasattr(model, "output_dim"):
        return int(getattr(model, "output_dim"))
    if hasattr(model, "config") and hasattr(model.config, "hidden_size"):
        return int(model.config.hidden_size)
    if hasattr(model, "model"):
        nested_model = getattr(model, "model")
        if hasattr(nested_model, "config") and hasattr(nested_model.config, "hidden_size"):
            return int(nested_model.config.hidden_size)
    raise ValueError("Unable to infer output dimension from the provided base model.")


def _to_device(batch: Mapping[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: v.to(device) for k, v in batch.items()}


class PairwiseMLPReranker(BaseRerankingModel):
    """Reranking model that augments a LEALLA encoder with an MLP head."""

    def __init__(
        self,
        model_name_or_path: str,
        *,
        state_dict_path: str | None = None,
        target_dim: int | None = None,
        output_type: ModelOutputType | str | None = None,
        tokenizer_name_or_path: str | None = None,
        mlp_hidden_dim: int | None = None,
        dropout: float = 0.1,
        device: torch.device | str = "cpu",
    ) -> None:
        super().__init__()
        self.device = torch.device(device)

        resolved_output_type = _maybe_convert_output_type(output_type)
        self.base_model = ModelFactory.auto_load_from_file(
            model_name_or_path,
            state_dict_path=state_dict_path,
            target_dim=target_dim,
            output_type=resolved_output_type,
        )
        self.base_model.eval()

        self.embedding_dim = _infer_output_dim(self.base_model)
        hidden_dim = mlp_hidden_dim or self.embedding_dim

        self.classifier = nn.Sequential(
            nn.Linear(self.embedding_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, 1),
        )

        tokenizer_id = tokenizer_name_or_path or model_name_or_path
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
        self.loss_fn = nn.BCEWithLogitsLoss()

        self.to(self.device)

    def forward(
        self,
        mention_tokens: Mapping[str, torch.Tensor],
        entity_tokens: Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        mention_tokens = _to_device(mention_tokens, self.device)
        entity_tokens = _to_device(entity_tokens, self.device)

        mention_embeddings = self._encode(mention_tokens)
        entity_embeddings = self._encode(entity_tokens)

        combined = torch.cat([mention_embeddings, entity_embeddings], dim=-1)
        logits = self.classifier(combined).squeeze(-1)
        return logits

    def train_step(self, data: Dict[str, Any]) -> torch.Tensor:
        self.train()

        mention_tokens = data["mention_tokens"]
        entity_tokens = data["entity_tokens"]
        labels = data["labels"].to(self.device).float().view(-1)

        logits = self.forward(mention_tokens, entity_tokens).view(-1)
        loss = self.loss_fn(logits, labels)
        return loss

    def train(self):
        super().train()
        # Make sure that base model is never trained.
        self.base_model.eval()

    @torch.inference_mode()
    def score(self, mention: str, entity_description: str) -> float:
        mention_tokens = self.tokenizer(
            mention,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)
        entity_tokens = self.tokenizer(
            entity_description,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)
        mention_tokens["attention_mask"] = create_attention_mask(mention_tokens["input_ids"])
        entity_tokens["attention_mask"] = create_attention_mask(entity_tokens["input_ids"])

        logits = self.forward(
            {k: v for k, v in mention_tokens.items()},
            {k: v for k, v in entity_tokens.items()},
        )
        probability = torch.sigmoid(logits).item()
        return probability

    @torch.inference_mode()
    def _encode(self, tokens: Mapping[str, torch.Tensor] | torch.Tensor) -> torch.Tensor:
        if isinstance(tokens, Mapping):
            input_ids = tokens["input_ids"]
            attention_mask = tokens.get("attention_mask")
            if attention_mask is None:
                attention_mask = create_attention_mask(input_ids)
        else:
            input_ids = tokens
            attention_mask = create_attention_mask(tokens)

        return self.base_model(input_ids=input_ids, attention_mask=attention_mask)
