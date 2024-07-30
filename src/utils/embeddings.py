""" Utils for embedding tokens.
"""

import logging
from pathlib import Path
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import BertModel

from utils.multifile_dataset import MultiFileDataset


def _create_attention_mask(toks, padding_value=0):
    return (toks != padding_value).long()


def embed(dataset, model, batch_size=(16384 * 4), return_tokens=False):
    """embeds dataset and returns embeddings, qids tuple.

    Note that all embeddings are held in memory at once which can consume a lot of RAM.
    TODO: We could implement some incremental saving.

    Args:
        dataset (Dataset/IterableDataset):
        model: Model for embedding the dataset, last pooling_layer is used
        batch_size (int, optional): Defaults to (16384 * 4).

    Returns:
        tuple[np.arr, np.arr]: Embs and qids.
    """
    model.eval()

    # We usually work with IterableDataset subclass so no multiprocessing
    data_loader = DataLoader(dataset, batch_size, num_workers=0)

    if torch.cuda.is_available():
        model = torch.nn.DataParallel(model).cuda()
    else:
        model = torch.nn.DataParallel(model)

    qids = []
    embeddings = []

    if return_tokens:
        tokens = []

    with torch.no_grad():
        for batch_toks, batch_qids in data_loader:
            batch_toks = batch_toks.to(torch.int64)
            attention_mask = _create_attention_mask(batch_toks)
            if torch.cuda.is_available():
                batch_toks = batch_toks.cuda()
                attention_mask = attention_mask.cuda()
            batch_embeddings = model(batch_toks, attention_mask).pooler_output
            batch_embeddings = batch_embeddings.cpu().numpy().astype(np.float16)
            qids.extend(batch_qids)
            embeddings.extend(batch_embeddings)
            if return_tokens:
                tokens.extend(batch_toks)
    if return_tokens:
        return np.array(embeddings), np.array(qids), np.array(tokens)
    return np.array(embeddings), np.array(qids)


def get_embs_and_qids(source_dir: Path, model: nn.Module, batch_size=16384):
    """
    Given a directory with (multiple) token_qid npz files embedds them with model and returns
    tuple (embs, qids) of two very large arrays.
    """
    dataset = MultiFileDataset(source_dir)
    embs, qids = embed(dataset, model, batch_size)
    return embs, qids


def embs_from_tokens_and_model_name(source, model_name, batch_size, dest):
    model = BertModel.from_pretrained(model_name)
    embs_from_tokens_and_model(source, model, batch_size, dest)


def embs_from_tokens_and_model(source, model, batch_size, dest):
    embs, qids = get_embs_and_qids(Path(source), model, batch_size)
    np.savez_compressed(f"{dest}/embs_qids.npz", embs=embs, qids=qids)
