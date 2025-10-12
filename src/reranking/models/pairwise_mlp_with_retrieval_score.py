from __future__ import annotations

from typing import Sequence

import torch

from reranking.models.pairwise_mlp import PairwiseMLPReranker
from utils.model_factory import ModelOutputType


class PairwiseMLPRerankerWithRetrievalScore(PairwiseMLPReranker):
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
    ) -> None:
        super().__init__(
            model_name_or_path=model_name_or_path,
            state_dict_path=state_dict_path,
            target_dim=target_dim,
            output_type=output_type,
            tokenizer_name_or_path=tokenizer_name_or_path,
            mlp_hidden_dim=mlp_hidden_dim,
            dropout=dropout,
        )

    @torch.inference_mode()
    def score(
        self, mention: str | Sequence[str], entity_description: str | Sequence[str]
    ) -> float | torch.Tensor:
        return super().score(mention, entity_description)

    @torch.inference_mode()
    def score_from_tokens(self, mentions: torch.Tensor, entities: torch.Tensor) -> float:
        logits, mention_embeddings, entity_embeddings = self.model.forward(
            mentions, entities, return_embeddings=True
        )
        probability1 = torch.sigmoid(logits)
        probability2 = torch.sigmoid(mention_embeddings @ entity_embeddings.T)
        print(probability1, probability2)
        probability = (probability1 + probability2.diagonal()) / 2
        return probability
