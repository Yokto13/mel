import logging
from pathlib import Path

import gin
import numpy as np
from torch.utils.data import IterableDataset
from utils.embeddings import embed_generator
from utils.multifile_dataset import MultiFileDataset

from finetunings.finetune_model.train import load_model

_logger = logging.getLogger("generate_epochs.embed_links_for_generation")


def _get_dataset(links_tokens_dir_path: str) -> IterableDataset:
    return MultiFileDataset(links_tokens_dir_path)


def _save(dest_dir_path: str, embs, qids, tokens, idx: int):
    np.savez_compressed(
        Path(dest_dir_path) / f"embs_qids_tokens_{idx}.npz",
        embs=embs,
        qids=qids,
        tokens=tokens,
    )


@gin.configurable
def embed_links_for_generation(
    links_tokens_dir_path: str,
    model_path: str,
    batch_size: int,
    dest_dir_path: str,
    state_dict_path: str,
    target_dim: int | None = None,
    output_type: str | None = None,
    per_save_size: int = 50,
) -> None:
    dataset = _get_dataset(links_tokens_dir_path)
    model = load_model(model_path, state_dict_path, target_dim, output_type)

    cummulative_embeddings = []
    cummulative_qids = []
    cummulative_tokens = []

    save_idx = 0

    for embs, qids, tokens in embed_generator(dataset, model, batch_size):
        cummulative_embeddings.extend(embs)
        cummulative_qids.extend(qids)
        cummulative_tokens.extend(tokens)
        save_idx += 1
        print(save_idx)
        if save_idx % per_save_size == 0:
            _save(
                dest_dir_path,
                np.array(cummulative_embeddings),
                np.array(cummulative_qids),
                np.array(cummulative_tokens),
                save_idx,
            )
            cummulative_embeddings = []
            cummulative_qids = []
            cummulative_tokens = []

    if len(cummulative_embeddings) > 0:
        _save(
            dest_dir_path,
            np.array(cummulative_embeddings),
            np.array(cummulative_qids),
            np.array(cummulative_tokens),
            save_idx,
        )
