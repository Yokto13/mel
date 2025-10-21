from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Mapping, Sequence

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


class PairwiseMLPRerankerWithLargeContextEmb(BaseRerankingModel):
    """Reranking model that augments a LEALLA encoder with an MLP head that uses paraphrase embedding to get more context."""

    def __init__(
        self,
        model_name_or_path: str,
        qid_to_paraphrase_emb: Dict[int, torch.Tensor],
        qid_to_base_emb: Dict[int, torch.Tensor],
        *,
        state_dict_path: str | None = None,
        target_dim: int | None = None,
        output_type: ModelOutputType | str | None = None,
        tokenizer_name_or_path: str | None = None,
        dropout: float = 0.1,
        ema_decay: float = 0.9999,
    ) -> None:
        super().__init__()
        self.ema_decay = ema_decay

        resolved_output_type = _maybe_convert_output_type(output_type)
        self.base_model = ModelFactory.auto_load_from_file(
            model_name_or_path,
            state_dict_path=state_dict_path,
            target_dim=target_dim,
            output_type=resolved_output_type,
        )
        self.base_model.eval()
        self.base_model.requires_grad_(False)

        self.paraphrase_model_embedding_dim = next(iter(qid_to_paraphrase_emb.values())).shape[0]
        self.base_model_embedding_dim = next(iter(qid_to_base_emb.values())).shape[0]

        hidden_dim = 2 * self.base_model_embedding_dim + self.paraphrase_model_embedding_dim

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(4 * hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, 1),
        )

        self.classifier_ema = deepcopy(self.classifier)

        self.model = _PairwiseMLPReranker(
            self.base_model, self.classifier, qid_to_paraphrase_emb, qid_to_base_emb
        )

        tokenizer_id = tokenizer_name_or_path or model_name_or_path
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(
        self,
        mention_tokens: torch.Tensor,
        entity_tokens: torch.Tensor,
    ) -> torch.Tensor:
        return self.model.forward(mention_tokens, entity_tokens)

    def train_step_imp(self, data: Dict[str, Any]) -> torch.Tensor:
        self.train()

        mention_tokens = data["mention_tokens"]
        labels = data["labels"].float().view(-1)
        qids = data["qids"].view(-1)

        logits = self.model.forward(mention_tokens, qids).view(-1)
        loss = self.loss_fn(logits, labels)
        return loss

    def update_ema(self) -> None:
        with torch.no_grad():
            for param, ema_param in zip(
                self.classifier.parameters(), self.classifier_ema.parameters()
            ):
                ema_param.data.mul_(self.ema_decay).add_(param.data, alpha=1 - self.ema_decay)

    def save(self, path: str) -> None:
        ema_path = path.replace(".pth", "_ema.pth")
        torch.save(self.classifier_ema.state_dict(), ema_path)
        torch.save(self.classifier.state_dict(), path)

    def load(self, path: str) -> None:
        state_dict = torch.load(path, map_location="cpu")
        self.classifier_ema.load_state_dict(state_dict)
        self.classifier.load_state_dict(state_dict)

    @torch.inference_mode()
    def score(
        self, mention: str | Sequence[str], entity_description: str | Sequence[str]
    ) -> float | torch.Tensor:
        single_pair = isinstance(mention, str) and isinstance(entity_description, str)

        if isinstance(mention, str):
            mention_batch = [mention]
        elif isinstance(mention, Sequence):
            mention_batch = list(mention)
        else:
            raise TypeError("Mentions must be a string or a sequence of strings.")

        if isinstance(entity_description, str):
            entity_batch = [entity_description]
        elif isinstance(entity_description, Sequence):
            entity_batch = list(entity_description)
        else:
            raise TypeError("Entity descriptions must be a string or a sequence of strings.")

        if len(mention_batch) != len(entity_batch):
            if len(mention_batch) == 1:
                mention_batch = mention_batch * len(entity_batch)
                single_pair = False
            elif len(entity_batch) == 1:
                entity_batch = entity_batch * len(mention_batch)
                single_pair = False
            else:
                raise ValueError(
                    "Mention and entity batches must be the same length or broadcastable."
                )

        mention_tokens = self.tokenizer(
            mention_batch,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )["input_ids"]
        entity_tokens = self.tokenizer(
            entity_batch,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )["input_ids"]

        probabilities = self.score_from_tokens(mention_tokens, entity_tokens)
        if not isinstance(probabilities, torch.Tensor):
            probabilities = torch.as_tensor(probabilities)

        probabilities = probabilities.reshape(-1).detach().cpu()

        if single_pair:
            return float(probabilities[0])
        return probabilities

    @torch.inference_mode()
    def score_from_tokens(self, mentions: torch.Tensor, qids: torch.Tensor) -> torch.Tensor:
        logits = self.model.forward(mentions, qids)
        probability = torch.sigmoid(logits).reshape(-1)
        return probability

    def classifier_forward(self, x):
        return self.model.classifier_forward(x)


class _PairwiseMLPReranker(nn.Module):
    def __init__(
        self,
        base_model: nn.Module,
        classifier: nn.Module,
        qid_to_paraphrase_emb: Dict[int, torch.Tensor],
        qid_to_base_emb: Dict[int, torch.Tensor],
    ) -> None:
        super().__init__()
        self.base_model = base_model
        self.classifier = classifier
        self.qid_to_paraphrase_emb = qid_to_paraphrase_emb
        self.qid_to_base_emb = qid_to_base_emb

    def forward(
        self,
        mention_tokens: torch.Tensor,
        qids: torch.Tensor,
        return_embeddings: bool = False,
    ) -> torch.Tensor:

        mention_embeddings = self._encode(mention_tokens)
        paraphrase_embeddings = torch.stack(
            [self.qid_to_paraphrase_emb[int(qid)] for qid in qids], dim=0
        ).to(mention_embeddings.device)
        base_embeddings = torch.stack([self.qid_to_base_emb[int(qid)] for qid in qids], dim=0).to(
            mention_embeddings.device
        )

        combined = torch.cat([mention_embeddings, paraphrase_embeddings, base_embeddings], dim=-1)
        logits = self.classifier(combined).squeeze(-1)
        if return_embeddings:
            return logits, mention_embeddings, paraphrase_embeddings, base_embeddings
        return logits

    def train(self, mode: bool = True) -> _PairwiseMLPReranker:
        super().train(mode)
        # Make sure that base model is never trained.
        self.base_model.eval()
        return self

    def classifier_forward(self, x):
        return self.classifier(x)

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
