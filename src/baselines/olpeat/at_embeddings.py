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

from pathlib import Path

import torch
import torch.nn as nn

from models.data.only_once_dataset import OnlyOnceDataset
from utils.embeddings import embed
from utils.multifile_dataset import MultiFileDataset


def get_embs_and_qids(source_dir: Path, model: nn.Module, batch_size=16384):
    """
    Given a directory with (multiple) token_qid npz files embedds them with model and returns
    tuple (embs, qids) of two very large arrays.
    """
    dataset = MultiFileDataset(source_dir)
    dataset = OnlyOnceDataset(dataset)
    embs, qids, tokens = embed(dataset, model, batch_size, return_tokens=True)

    searcher = get_searcher(tokens, embs)

    dataset = MultiFileDataset(source_dir)


def get_searcher(tokens, metadata):
    return TokensSearcher.
    
