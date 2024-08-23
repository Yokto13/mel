import logging
from pathlib import Path

import numpy as np
from torch.utils.data import IterableDataset
import torch.nn as nn

from utils.model_factory import ModelFactory
from utils.multifile_dataset import MultiFileDataset
from utils.embeddings import embed

_logger = logging.getLogger("generate_epochs.embed_links_for_generation")


def _load_model(model_path: str, state_dict_path: str | None) -> nn.Module:
    if state_dict_path is None:
        return ModelFactory.load_bert_from_file(model_path)
    else:
        return ModelFactory.load_bert_from_file_and_state_dict(
            model_path, state_dict_path
        )


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
) -> None:
    dataset = _get_dataset(links_tokens_dir_path)
    model = _load_model(model_path, state_dict_path)

    embs, qids, tokens = embed(
        dataset, model, batch_size, return_qids=True, return_tokens=True
    )

    _save(dest_dir_path, embs, qids, tokens)
