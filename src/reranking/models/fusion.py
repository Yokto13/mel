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


class SimpleCrossAttentionLayer(nn.Module):
    """A simple wrapper for PyTorch's MultiheadAttention to perform cross-attention."""

    def __init__(self, query_dim, key_value_dim, num_heads=8, dropout=0.1):
        super().__init__()
        # The MHA layer that handles everything, including projections!
        # Note: We must set batch_first=True for BERT-style inputs.
        self.attention = nn.MultiheadAttention(
            embed_dim=query_dim,
            num_heads=num_heads,
            kdim=key_value_dim,
            vdim=key_value_dim,
            dropout=dropout,
            batch_first=True,  # Important for [batch, seq_len, dim] tensors
        )
        self.norm = nn.LayerNorm(query_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key_value, attention_mask=None):
        # query: The tensor that asks the questions (from the target model).
        # key_value: The tensor that provides the context (from the source model).

        # The MHA layer returns the attention output and weights. We only need the output.
        attn_output, _ = self.attention(
            query=query,
            key=key_value,
            value=key_value,
            key_padding_mask=attention_mask,  # Optional mask for padded tokens
        )

        # Add residual connection and layer norm
        output = self.norm(query + self.dropout(attn_output))
        return output


class BertFusionModel(nn.Module):
    def __init__(self, model1, model2):
        super().__init__()
        self.base = model1  # The 24-layer model (dim 192)
        self.paraphrase = model2  # The 12-layer model (dim 384)

        # Simpler cross-attention layers using the wrapper
        self.cross_layers_1_to_2 = nn.ModuleList(
            [SimpleCrossAttentionLayer(query_dim=384, key_value_dim=192) for _ in range(12)]
        )
        self.cross_layers_2_to_1 = nn.ModuleList(
            [SimpleCrossAttentionLayer(query_dim=192, key_value_dim=384) for _ in range(12)]
        )

    def forward(self, input_ids1, attention_mask1, input_ids2, attention_mask2):
        # 1. Get initial embeddings
        hidden_states1 = self.base.embeddings(input_ids=input_ids1)
        hidden_states2 = self.paraphrase.embeddings(input_ids=input_ids2)

        # (Note: You may need to adapt the huggingface attention_mask format for MHA's key_padding_mask)

        # 2. Iteratively process through layers with fusion
        for i in range(12):
            # Process layers for each model
            hidden_states1 = self.base.encoder.layer[2 * i](hidden_states1)[0]
            hidden_states1 = self.base.encoder.layer[2 * i + 1](hidden_states1)[0]
            hidden_states2 = self.paraphrase.encoder.layer[i](hidden_states2)[0]

            # Store states before the cross-attention step to avoid in-place modification issues
            temp_states1, temp_states2 = hidden_states1, hidden_states2

            # Cross-attention exchange
            hidden_states1 = self.cross_layers_2_to_1[i](query=temp_states1, key_value=temp_states2)
            hidden_states2 = self.cross_layers_1_to_2[i](query=temp_states2, key_value=temp_states1)

        # 3. Get final outputs
        pooled_output1 = self.base.pooler(hidden_states1)
        pooled_output2 = self.paraphrase.pooler(hidden_states2)

        return torch.cat([pooled_output1, pooled_output2], dim=-1)


class _Model(nn.Module):
    def __init__(self, fusion_model: BertFusionModel, dropout: float = 0.1) -> None:
        super().__init__()
        self.fusion_model = fusion_model
        self.embedding_dim = 384 + 192
        self.gpt_layer = GPTLayer(model_width=self.embedding_dim, dropout=dropout)
        self.final_layer = nn.Linear(self.embedding_dim, 1)

    def forward(
        self,
        input_ids1: torch.Tensor,
        attention_mask1: torch.Tensor,
        input_ids2: torch.Tensor,
        attention_mask2: torch.Tensor,
    ) -> torch.Tensor:
        # print(ids.shape, attention_mask.shape)
        base_embeddings = self.fusion_model(
            input_ids1, attention_mask1, input_ids2, attention_mask2
        )
        # print("base_embeddings", base_embeddings.shape)
        x = self.gpt_layer(base_embeddings)
        # print("x", x.shape)
        logits = self.final_layer(x).squeeze(-1)
        # print("logits", logits.shape)
        return logits


class FusionReranker(BaseRerankingModel):
    """Reranking model that augments a LEALLA encoder with an MLP head."""

    def __init__(
        self,
        base_model_name_or_path: str,
        paraphrase_model_name_or_path: str,
        qid_to_para_toks: Dict[str, torch.Tensor],
        *,
        base_state_dict_path: str | None = None,
        tokenizer_name_or_path: str | None = None,
        dropout: float = 0.1,
        ema_decay: float = 0.9999,
    ) -> None:
        super().__init__()

        self.qid_to_para_toks = qid_to_para_toks

        self.ema_decay = ema_decay

        self.base_model = ModelFactory.auto_load_from_file(
            base_model_name_or_path,
            state_dict_path=base_state_dict_path,
        ).model
        self.paraphrase_model = ModelFactory.auto_load_from_file(
            paraphrase_model_name_or_path,
            output_type="sentence_transformer",
        ).model

        fusion_model = BertFusionModel(self.base_model, self.paraphrase_model)

        self.model = _Model(
            fusion_model=fusion_model,
            dropout=dropout,
        )
        self.model_ema = deepcopy(self.model)

        tokenizer_id = tokenizer_name_or_path or base_model_name_or_path
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(
        self,
        mention_tokens: torch.Tensor,
        entity_tokens: torch.Tensor,
    ) -> torch.Tensor:
        ids, attention_mask = self.prepare_for_forward(mention_tokens, entity_tokens)
        return self.model(ids, attention_mask)

    def prepare_for_forward(self, mention_tokens: torch.Tensor, entity_tokens: torch.Tensor):
        ids = rearrange([mention_tokens, entity_tokens], "d b n -> b (d n)")
        attention_mask = create_attention_mask(ids).to(dtype=ids.dtype, device=ids.device)
        return ids, attention_mask

    def train_step_imp(self, data: Dict[str, Any]) -> torch.Tensor:
        self.train()

        mention_tokens = data["mention_tokens"]
        entity_tokens = data["entity_tokens"]
        labels = data["labels"].float()
        qids = data["qids"].numpy()

        para_tokens = torch.stack([self.qid_to_para_toks[qid] for qid in qids], dim=0).to(
            device=mention_tokens.device
        )
        para_attention_mask = create_attention_mask(para_tokens).to(
            dtype=para_tokens.dtype, device=para_tokens.device
        )

        ids, attention_mask = self.prepare_for_forward(mention_tokens, entity_tokens)
        attention_mask = attention_mask.to(dtype=ids.dtype, device=ids.device)
        logits = self.model(ids, attention_mask, para_tokens, para_attention_mask)

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
        raise NotImplementedError("Use score_from_tokens_and_qids for FusionReranker.")

    @torch.inference_mode()
    def score_from_tokens_and_qids(
        self, mentions: torch.Tensor, entities: torch.Tensor, qids: torch.Tensor
    ) -> torch.Tensor:
        qids = qids.numpy()
        para_tokens = torch.stack([self.qid_to_para_toks[qid] for qid in qids], dim=0).to(
            device=mentions.device
        )
        para_attention_mask = create_attention_mask(para_tokens).to(
            dtype=para_tokens.dtype, device=para_tokens.device
        )
        ids, attention_mask = self.prepare_for_forward(mentions, entities)
        attention_mask = attention_mask.to(dtype=ids.dtype, device=ids.device)
        logits = self.model(ids, attention_mask, para_tokens, para_attention_mask)

        probabilities = torch.sigmoid(logits).reshape(-1)

        if not isinstance(probabilities, torch.Tensor):
            probabilities = torch.as_tensor(probabilities)

        probabilities = probabilities.reshape(-1).detach().cpu()
        return probabilities

    @torch.inference_mode()
    def score_from_tokens(self, mentions: torch.Tensor, entities: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Use score_from_tokens_and_qids for FusionReranker.")
