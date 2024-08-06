import logging
from pathlib import Path

import numpy as np
import torch
from transformers import BertModel

from utils.multifile_dataset import MultiFileDataset
from utils.embeddings import embed

_logger = logging.getLogger("generate_epochs.embed_links_for_generation")


def embed_links_for_generation(
    links_tokens_dir: str,
    model_path: str,
    batch_size: int,
    dest_dir: str,
    state_dict_path: str,
) -> None:
    model = BertModel.from_pretrained(model_path)
    dataset = MultiFileDataset(links_tokens_dir)

    if state_dict_path is not None:
        _logger.debug("Loading model state dict")
        d = torch.load(state_dict_path)
        model.load_state_dict(d)
    embs, qids, tokens = embed(
        dataset, model, batch_size, return_qids=True, return_tokens=True
    )

    np.savez_compressed(
        Path(dest_dir) / "embs_qids_tokens.npz", embs=embs, qids=qids, tokens=tokens
    )
