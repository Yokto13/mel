"""Produces embeddings suitable for alias table.

Currently, we are sourcing aliases from links, thus they are likely to repeat.
For example, there can be thousands of links for the alias "Paris".
Therefore, we need to run "set" over all these aliases.
Previously, this was done post embedding phase which wasted resources, now OnlyOnceDataset makes sure
that we infere embeddings only once.

The results are saved to npz format.
There are 3 values per line:
    - embedding
    - qid
    - count: the number of occurrences of this particular alias. This is needed in OLPEAT to correctly choose
            which entity to consider.
"""

from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from transformers import BertModel

from models.data.only_once_dataset import OnlyOnceDataset
from models.data.tokens_multiplicity_counter import TokensMultiplicityCounter
from models.data.tokens_searcher import TokensSearcher
from utils.embeddings import embed
from utils.multifile_dataset import MultiFileDataset


def embed_for_at_to(
    source_dir: Path | str, model: nn.Module, output_dir: Path | str, batch_size=16384
):
    """
    Given a directory with (multiple) token_qid npz files embedds them with model and returns
    tuple (embs, qids) of two very large arrays.
    """
    if type(output_dir) == str:
        output_dir = Path(output_dir)
    if type(source_dir) == str:
        source_dir = Path(source_dir)

    dataset = MultiFileDataset(source_dir)
    dataset = OnlyOnceDataset(dataset)
    print("embedding...")
    embs, tokens = embed(
        dataset, model, batch_size, return_tokens=True, return_qids=False
    )
    print("embedded")

    dataset = MultiFileDataset(source_dir)
    print("Building counter...")
    counter = _get_multiplicity_counter(dataset)
    print("counter built")

    print("building searcher...")
    searcher = {tuple(tok): emb for tok, emb in zip(tokens, embs)}
    print("searcher built")

    dataset = MultiFileDataset(source_dir)
    embs, qids, multiplicities = _map_dataset_to_result(
        dataset, searcher, counter, len(embs[0])
    )

    np.savez_compressed(
        output_dir / "embs_qids_counts", embs=embs, qids=qids, counts=multiplicities
    )


def _map_dataset_to_result(
    dataset, searcher: TokensSearcher, counter: TokensMultiplicityCounter, embs_size
):
    item_count = len(dataset)

    embs = np.empty((item_count, embs_size), dtype=np.float16)
    qids = np.empty((item_count, embs_size), dtype=np.int32)
    multiplicities = np.empty((item_count, embs_size), dtype=np.int32)

    for i, (toks, qid) in enumerate(dataset):
        embs[i] = searcher[tuple(toks)]
        qids[i] = qid
        multiplicities[i] = counter[qid][tuple(toks)]
    return embs, qids, multiplicities


def _get_multiplicity_counter(dataset):
    counter = {}
    for toks, qid in dataset:
        if qid not in counter:
            counter[qid] = defaultdict(int)
        counter[qid][tuple(toks)] += 1
    return counter


def embs_from_tokens_and_model_name_at(source, model_name, batch_size, dest):
    model = BertModel.from_pretrained(model_name)
    embed_for_at_to(source, model, dest, batch_size)
