import torch
from transformers.utils import ModelOutput


class ChangeDimWrapper(torch.nn.Module):
    def __init__(self, embedding_model: torch.nn.Module, target_dim: int):
        super().__init__()
        self.embedding_model = embedding_model
        self.target_dim = target_dim
        self.embedding_dim = embedding_model.output_dim
        self.mapping = torch.nn.Linear(self.embedding_dim, self.target_dim)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor) -> ModelOutput:
        embeddings = self.embedding_model(x, attention_mask=attention_mask)
        mapped_embeddings = self.mapping(embeddings)
        return mapped_embeddings