from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict

import torch
from torch import nn


class BaseRerankingModel(nn.Module, ABC):
    """Abstract base class for reranking models."""

    @abstractmethod
    def train_step(self, data: Dict[str, Any]) -> torch.Tensor:
        """Run a single training step on the provided batch data and return the loss."""
        raise NotImplementedError

    @abstractmethod
    def score(self, mention: str, entity_description: str) -> float:
        """Compute a similarity-based probability that the mention refers to the entity."""
        raise NotImplementedError
