from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Sequence

import torch
from einops import rearrange
from torch import nn
from transformers import AutoTokenizer

from reranking.models.base import BaseRerankingModel
from reranking.models.context_emb_with_attention import GPTLayer
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


class _Model(nn.Module):
    def __init__(
        self, base_model: nn.Module, embedding_dim: int, dropout: float, output_size: int
    ) -> None:
        super().__init__()
        self.base_model = base_model
        self.embedding_dim = embedding_dim
        self.gpt_layer = GPTLayer(model_width=embedding_dim, dropout=dropout)
        self.final_layer = nn.Linear(embedding_dim, output_size)

    def forward(self, ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        # print(ids.sh1ape, attention_mask.shape)
        base_embeddings = self.base_model(ids, attention_mask)
        # print("base_embeddings", base_embeddings.shape)
        x = self.gpt_layer(base_embeddings)
        # print("x", x.shape)
        return self.final_layer(x)


class FullLEALLARerankerMulticlass(BaseRerankingModel):
    """Reranking model that augments a LEALLA encoder with an MLP head."""

    def __init__(
        self,
        model_name_or_path: str,
        *,
        state_dict_path: str | None = None,
        tokenizer_name_or_path: str | None = None,
        dropout: float = 0.1,
        ema_decay: float = 0.9999,
        embedding_dim: int | None = None,
        output_size: int = 7,
    ) -> None:
        super().__init__()

        self.ema_decay = ema_decay

        self.base_model = ModelFactory.auto_load_from_file(
            model_name_or_path,
            state_dict_path=state_dict_path,
        )
        if embedding_dim is None:
            self.embedding_dim = _infer_output_dim(self.base_model)
        else:
            self.embedding_dim = embedding_dim

        self.model = _Model(
            base_model=self.base_model,
            embedding_dim=self.embedding_dim,
            dropout=dropout,
            output_size=output_size,
        )
        self.model_ema = deepcopy(self.model)

        tokenizer_id = tokenizer_name_or_path or model_name_or_path
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
        self.loss_fn = nn.CrossEntropyLoss()

        self.tokenizer = AutoTokenizer.from_pretrained(
            "/lnet/work/home-students-external/farhan/troja/outputs/models/LEALLA-base"
        )

    def forward(
        self,
        mention_tokens: torch.Tensor,
        entity_tokens: torch.Tensor,
    ) -> torch.Tensor:
        ids, attention_mask = self.prepare_for_forward(mention_tokens, entity_tokens)
        return self.model(ids, attention_mask)

    def prepare_for_forward(self, mention_tokens: torch.Tensor, entity_tokens: torch.Tensor):
        entity_tokens = torch.cat(entity_tokens, dim=1)
        ids = torch.cat([mention_tokens, entity_tokens], dim=1)
        attention_mask = create_attention_mask(ids).to(dtype=ids.dtype, device=ids.device)
        return ids, attention_mask

    def train_step_imp(self, data: Dict[str, Any]) -> torch.Tensor:
        self.train()

        mention_tokens = data["mention_tokens"]
        entity_tokens = data["entity_tokens"]
        labels = data["labels"].float()
        ids, attention_mask = self.prepare_for_forward(mention_tokens, entity_tokens)

        logits = self.model(ids, attention_mask)

        loss = self.loss_fn(logits, labels)
        return loss

    def update_ema(self) -> None:
        with torch.no_grad():
            for param, ema_param in zip(self.model.parameters(), self.model_ema.parameters()):
                ema_param.data.mul_(self.ema_decay).add_(param.data, alpha=1 - self.ema_decay)

    def save(self, path: str) -> None:
        ema_path = path.replace(".pth", "_ema.pth")
        torch.save(self.model_ema.state_dict(), ema_path)
        torch.save(self.model.state_dict(), path)

    def load(self, path: str) -> None:
        state_dict = torch.load(path, map_location="cpu")
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("_orig_mod.module.model."):
                new_k = k.replace("_orig_mod.module.model.", "")
            elif k.startswith("module."):
                new_k = k.replace("module.", "")
            else:
                new_k = k
            new_state_dict[new_k] = v
        state_dict = new_state_dict
        self.model_ema.load_state_dict(state_dict)
        self.model.load_state_dict(state_dict)

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
    def score_from_tokens(self, mentions: torch.Tensor, entities: torch.Tensor) -> torch.Tensor:
        ids, attention_mask = self.prepare_for_forward(mentions, entities)
        logits = self.model(ids, attention_mask)
        probability = torch.sigmoid(logits).reshape(-1)
        return probability
