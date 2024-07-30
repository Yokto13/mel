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
from models.data.tokens_multiplicity_counter import TokensMultiplicityCounter
from models.data.tokens_searcher import TokensSearcher
from utils.embeddings import embed
from utils.multifile_dataset import MultiFileDataset


def embed_for_at_to(
    source_dir: Path, model: nn.Module, output_dir: Path, batch_size=16384
):
    """
    Given a directory with (multiple) token_qid npz files embedds them with model and returns
    tuple (embs, qids) of two very large arrays.
    """
    dataset = MultiFileDataset(source_dir)
    dataset = OnlyOnceDataset(dataset)
    embs, tokens = embed(
        dataset, model, batch_size, return_tokens=True, return_qids=False
    )

    searcher = TokensSearcher(tokens, embs)

    dataset = MultiFileDataset(source_dir)


def _map_dataset_to_embeddings(datset, searcher):
    pass


def _get_multiplicity_counter(dataset):
    counter = TokensMultiplicityCounter()
    for x in dataset:
        counter
