from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Sequence

import torch
from torch import nn


class BaseRerankingModel(nn.Module, ABC):
    """Abstract base class for reranking models."""

    def train_step(self, data: Dict[str, Any]) -> torch.Tensor:
        """Run a single training step on the provided batch data and return the loss."""
        self.update_ema()
        return self.train_step_imp(data)

    @abstractmethod
    def train_step_imp(self, data: Dict[str, Any]) -> torch.Tensor:
        """Run a single training step on the provided batch data and return the loss."""
        raise NotImplementedError

    @abstractmethod
    def update_ema(self) -> None:
        """Update the EMA (Exponential Moving Average) of model parameters."""
        raise NotImplementedError

    @abstractmethod
    def score(
        self, mention: str | Sequence[str], entity_description: str | Sequence[str]
    ) -> torch.Tensor | float:
        """Compute similarity-based probability for one or more mention/entity pairs."""
        raise NotImplementedError

    @abstractmethod
    def score_from_tokens(self, mention: Any, entity_description: Any) -> torch.Tensor:
        """Compute similarity-based probabilities for tokenized mention/entity pairs."""
        raise NotImplementedError

    @abstractmethod
    def save(self, path: str) -> None:
        """Save the model to the specified path."""
        raise NotImplementedError

    @abstractmethod
    def load(self, path: str) -> None:
        """Load the model from the specified path."""
        raise NotImplementedError
