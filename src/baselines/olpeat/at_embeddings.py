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

import gin
import numpy as np
import torch.nn as nn

from models.data.only_once_dataset import OnlyOnceDataset
from utils.embeddings import embed
from utils.model_builder import ModelBuilder, ModelOutputType
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

    np.savez_compressed(output_dir / "embs_tokens.npz", embs=embs, tokens=tokens)


@gin.configurable
def embs_from_tokens_and_model_name_at(
    source, model_name, batch_size, dest, output_type: str
):
    print("Building model...")
    output_type = ModelOutputType(output_type)
    builder = ModelBuilder(model_name)
    builder.set_output_type(output_type)
    model = builder.build()

    embed_for_at_to(source, model, dest, batch_size)
