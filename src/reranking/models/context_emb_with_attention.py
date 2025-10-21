from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Mapping, Sequence

import torch
from einops import rearrange
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


class GPTLayer(nn.Module):
    def __init__(self, model_width: int, dropout: float) -> None:
        super().__init__()
        self.linear1 = nn.Linear(model_width, model_width * 4)
        self.activation = nn.GELU()
        self.linear2 = nn.Linear(model_width * 4, model_width)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        x = self.dropout(x)
        return x


class _CATransformerBlock(nn.Module):
    def __init__(self, model_width: int, dropout: float, num_heads: int) -> None:
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(model_width)
        self.self_attention = nn.MultiheadAttention(
            embed_dim=model_width, num_heads=num_heads, dropout=dropout
        )
        self.layer_norm2 = nn.LayerNorm(model_width)
        self.feed_forward = GPTLayer(model_width, dropout)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        # Self-attention block
        residual = x
        x = self.layer_norm1(x)
        x, _ = self.self_attention(x, context, x)
        x = x + residual

        # Feed-forward block
        residual = x
        x = self.layer_norm2(x)
        x = self.feed_forward(x)
        x = x + residual

        return x


class _DoubleCrossAttention(nn.Module):
    def __init__(self, model_width: int, dropout: float, num_heads: int) -> None:
        super().__init__()
        self.transformer1 = _CATransformerBlock(model_width, dropout, num_heads)
        self.transformer2 = _CATransformerBlock(model_width, dropout, num_heads)

    def forward(
        self, main_input: torch.Tensor, context_input1: torch.Tensor, context_input2: torch.Tensor
    ) -> torch.Tensor:
        x = self.transformer1(main_input, context_input1)
        x = self.transformer2(x, context_input2)
        return x


class _Classifier(nn.Module):
    def __init__(
        self,
        model_width: int,
        dropout: float,
        num_heads: int,
        num_layers: int,
        n_tokens: int,
        base_dim: int,
        paraphrase_dim: int,
    ) -> None:
        super().__init__()
        self.double_cross_attention = nn.ModuleList(
            [_DoubleCrossAttention(model_width, dropout, num_heads) for _ in range(num_layers)]
        )
        self.mewsli_to_tokens = nn.Linear(base_dim, model_width * n_tokens)
        self.base_to_tokens = nn.Linear(base_dim, model_width * n_tokens)
        self.paraphrase_to_tokens = nn.Linear(paraphrase_dim, model_width * n_tokens)

        self.final_projection = nn.Linear(model_width * n_tokens, 1)
        self.model_width = model_width

    def forward(
        self, mewsli_embs: torch.Tensor, base_embs: torch.Tensor, paraphrase_embs: torch.Tensor
    ) -> torch.Tensor:
        mewsli_tokens = self.mewsli_to_tokens(mewsli_embs)
        base_tokens = self.base_to_tokens(base_embs)
        paraphrase_tokens = self.paraphrase_to_tokens(paraphrase_embs)

        mewsli_tokens = rearrange(mewsli_tokens, "b (n d) -> n b d", n=self.model_width)
        base_tokens = rearrange(base_tokens, "b (n d) -> n b d", n=self.model_width)
        paraphrase_tokens = rearrange(paraphrase_tokens, "b (n d) -> n b d", n=self.model_width)

        for layer in self.double_cross_attention:
            mewsli_tokens = layer(mewsli_tokens, base_tokens, paraphrase_tokens)

        mewsli_tokens = rearrange(mewsli_tokens, "n b d -> b (n d)")
        logits = self.final_projection(mewsli_tokens).squeeze(-1)
        return logits


class ContextEmbWithAttention(BaseRerankingModel):
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
        model_width: int = 64,
        num_heads: int = 16,
        num_layers: int = 2,
        n_tokens: int = 64,
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

        self.classifier = _Classifier(
            model_width=model_width,
            dropout=dropout,
            num_heads=num_heads,
            num_layers=num_layers,
            n_tokens=n_tokens,
            base_dim=self.base_model_embedding_dim,
            paraphrase_dim=self.paraphrase_model_embedding_dim,
        )
        self.classifier.to(dtype=torch.float16)

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
        return self.model(mention_tokens, entity_tokens)

    def train_step_imp(self, data: Dict[str, Any]) -> torch.Tensor:
        self.train()

        mention_tokens = data["mention_tokens"]
        labels = data["labels"].float().view(-1)
        qids = data["qids"].view(-1)

        logits = self.model(mention_tokens, qids).view(-1)
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
        assert False, "Not correct, TODO FIX this"
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
        logits = self.model(mentions, qids)
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

        mention_embeddings = self._encode(mention_tokens).clone().detach()
        paraphrase_embeddings = torch.stack(
            [self.qid_to_paraphrase_emb[int(qid)] for qid in qids], dim=0
        ).to(mention_embeddings.device)
        base_embeddings = torch.stack([self.qid_to_base_emb[int(qid)] for qid in qids], dim=0).to(
            mention_embeddings.device
        )

        logits = self.classifier(
            mention_embeddings, base_embeddings, paraphrase_embeddings
        ).squeeze(-1)
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
