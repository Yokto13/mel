""" Utils for embedding tokens.
"""
import logging
from pathlib import Path
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
from transformers import BertModel


def _load_tokens(source_dir: Path):
    data_len = 0
    tok_dim = 0
    for file in tqdm(source_dir.iterdir(), desc="Counting toks and qids."):
        if not file.name.endswith("npz"):
            continue
        arrs = np.load(file)
        toks = arrs["tokens"]
        data_len += toks.shape[0]
        tok_dim = toks.shape[1]

    tokens = np.empty((data_len, tok_dim), dtype=np.uint16)
    qids = np.empty(data_len, dtype=np.uint32)
    idx = 0
    for file in tqdm(source_dir.iterdir(), desc="Loading toks and qids."):
        if not file.name.endswith("npz"):
            continue
        arrs = np.load(file)
        toks = arrs["tokens"]
        tokens[idx : idx + len(toks)] = toks
        qids[idx : idx + len(toks)] = arrs["qids"]
        idx += len(toks)
    return tokens, qids


def _create_attention_mask(toks, padding_value=0):
    return (toks != padding_value).long()


def _embed(toks, model, batch_size=(16384 * 4)):
    model.eval()

    toks = torch.tensor(toks.astype(np.int64), dtype=torch.long)

    out_size = model.config.hidden_size

    if torch.cuda.is_available():
        model = torch.nn.DataParallel(model).cuda()
    else:
        model = torch.nn.DataParallel(model)

    with torch.no_grad():
        # num_batches = (len(toks) + batch_size - 1) // batch_size

        embeddings = np.zeros((len(toks), out_size), np.float16)
        for i in tqdm(range(0, len(toks), batch_size), desc="Embedding tokens"):
            batch_toks = toks[i : i + batch_size]
            attention_mask = _create_attention_mask(batch_toks)
            if torch.cuda.is_available():
                batch_toks = batch_toks.cuda()
                attention_mask = attention_mask.cuda()
            outputs = model(batch_toks, attention_mask).pooler_output
            embeddings[i : i + len(batch_toks)] = (
                outputs.cpu().numpy().astype(np.float16)
            )

    return embeddings


def get_embs_and_qids(source_dir: Path, model: nn.Module, batch_size=16384):
    """
    Given a directory with (multiple) token_qid npz files embedds them with model and returns
    tuple (embs, qids) of two very large arrays.

    TODO: This might easily be a problem if there is a large number of tokens and they do not fit in RAM.
    """
    toks, qids = _load_tokens(source_dir)
    embs = _embed(toks, model, batch_size)
    return embs, qids


def embs_from_tokens_and_model_name(source, model_name, batch_size, dest):
    model = BertModel.from_pretrained(model_name)
    embs_from_tokens_and_model(source, model, batch_size, dest)


def embs_from_tokens_and_model(source, model, batch_size, dest):
    embs, qids = get_embs_and_qids(
        Path(source), model, batch_size
    )
    np.savez_compressed(f"{dest}/embs_qids.npz", embs=embs, qids=qids)