import torch
from transformers.utils import ModelOutput


class SmallerDim(torch.nn.Module):
    def __init__(self, embedding_model: torch.nn.Module, target_dim: int):
        super().__init__()
        self.embedding_model = embedding_model
        self.target_dim = target_dim
        self.embedding_dim = embedding_model.pooler.dense.out_features
        self.mapping = torch.nn.Linear(self.embedding_dim, self.target_dim)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor) -> ModelOutput:
        embeddings = self.embedding_model(
            x, attention_mask=attention_mask
        ).pooler_output
        mapped_embeddings = self.mapping(embeddings)
        return ModelOutput(pooler_output=mapped_embeddings)
