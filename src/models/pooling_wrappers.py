import torch
import torch.nn as nn


class PoolingWrapper(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        raise NotImplementedError

    @property
    def output_dim(self):
        return self.model.pooler.dense.out_features


class CLSWrapper(PoolingWrapper):
    def __init__(self, model: nn.Module):
        super().__init__(model)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        return self.model(input_ids, attention_mask).pooler_output


class SentenceTransformerWrapper(PoolingWrapper):
    def __init__(self, model: nn.Module):
        super().__init__(model)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        model_output = self.model(input_ids, attention_mask, return_dict=True)
        embeddings = self._mean_pooling(model_output, attention_mask)
        return torch.nn.functional.normalize(embeddings, p=2, dim=1)

    def _mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output.last_hidden_state
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )
