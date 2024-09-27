import logging
from pathlib import Path

import numpy as np
import torch.nn as nn
from torch.utils.data import IterableDataset
from utils.embeddings import embed
from utils.multifile_dataset import MultiFileDataset

from finetunings.finetune_model.train import load_model

_logger = logging.getLogger("generate_epochs.embed_links_for_generation")


def _get_dataset(links_tokens_dir_path: str) -> IterableDataset:
    return MultiFileDataset(links_tokens_dir_path)


def _save(dest_dir_path: str, embs, qids, tokens):
    np.savez_compressed(
        Path(dest_dir_path) / "embs_qids_tokens.npz",
        embs=embs,
        qids=qids,
        tokens=tokens,
    )


def embed_links_for_generation(
    links_tokens_dir_path: str,
    model_path: str,
    batch_size: int,
    dest_dir_path: str,
    state_dict_path: str,
    target_dim: int | None = None,
) -> None:
    dataset = _get_dataset(links_tokens_dir_path)
    model = load_model(model_path, state_dict_path, target_dim)

    embs, qids, tokens = embed(
        dataset, model, batch_size, return_qids=True, return_tokens=True
    )

    _save(dest_dir_path, embs, qids, tokens)
