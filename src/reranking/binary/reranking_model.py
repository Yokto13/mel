import torch
import torch.nn as nn
from transformers import AutoModel


class BinaryReranker(nn.Module):
    def __init__(
        self,
        transformer: AutoModel,
    ):
        super().__init__()
        self.transformer = transformer

        hidden_size = transformer.config.hidden_size
        self.final_layer = nn.Linear(hidden_size, 1)

    def forward(
        self,
        input_toks: torch.Tensor,
        input_attn: torch.Tensor,
    ) -> torch.Tensor:
        outputs = self.transformer(
            input_ids=input_toks,
            attention_mask=input_attn,
        )
        hidden_states = outputs.last_hidden_state
        logits = self.final_layer(hidden_states[:, 0, :])
        probs = torch.sigmoid(logits)
        return probs
