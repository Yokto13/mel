import functools
import os
from pathlib import Path


import gin

import numpy as np
import pandas as pd

from utils.qids_remap import remap_qids_decorator


current_file_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))
config_path = os.path.join(project_root, "configs", "general.gin")
gin.parse_config_file(config_path)


def _sort_by_output(output_idx: int):
    def _sort_by_output_wrapper(wrapped):
        @functools.wraps(wrapped)
        def _wrapper(*args, **kwargs):
            output: tuple[np.ndarray, ...] = wrapped(*args, **kwargs)
            sort_indices = np.argsort(output[output_idx], kind="stable")
            return [o[sort_indices] for o in output]

        return _wrapper

    return _sort_by_output_wrapper


# @_sort_by_output(1)
@remap_qids_decorator(qids_index=1, json_path=gin.REQUIRED)
def load_embs_and_qids(dir_path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Loads embeddings and qids from the directory.

    This should be the preferable than directly loading them to ensure that the logic of loading is just in one place.

    Args:
        dir_path (str | Path)

    Returns:
        tuple[np.ndarray, np.ndarray]: embs, qids
    """
    if type(dir_path) == str:
        dir_path = Path(dir_path)
    d = np.load(dir_path / "embs_qids.npz")
    return d["embs"], d["qids"]


# @_sort_by_output(1)
@remap_qids_decorator(qids_index=1, json_path=gin.REQUIRED)
def load_embs_qids_tokens(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Loads embeddings, qids, and tokens from the directory.

    This should be the preferable than directly loading them to ensure that the logic of loading is just in one place.

    Args:
        path (str | Path)

    Returns:
        tuple[np.ndarray, np.ndarray]: embs, qids, tokens
    """
    if type(path) == str:
        path = Path(path)
    is_dir = path.is_dir()
    if is_dir:
        d = np.load(path / "embs_qids_tokens.npz")
    else:
        d = np.load(path)
    return d["embs"], d["qids"], d["tokens"]


# @_sort_by_output(1)
@remap_qids_decorator(qids_index=1, json_path=gin.REQUIRED)
def load_mentions(file_path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    if type(file_path) == str:
        file_path = Path(file_path)
    d = np.load(file_path)
    return d["tokens"], d["qids"]


@remap_qids_decorator(qids_index=None, json_path=gin.REQUIRED)
def load_qids(file_path: str | Path) -> np.ndarray:
    if type(file_path) == str:
        file_path = Path(file_path)
    d = np.load(file_path)
    return d["qids"]


@_sort_by_output(1)
@remap_qids_decorator(qids_index=1, json_path=gin.REQUIRED)
def load_mentions_from_dir(dir_path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    tokens, qids = [], []
    for file in dir_path.iterdir():
        if file.is_file() and file.suffix == ".npz":
            d = np.load(file)
            tokens.extend(d["tokens"])
            qids.extend(d["qids"])
    return np.array(tokens), np.array(qids)


@remap_qids_decorator(qids_index=1, json_path=gin.REQUIRED)
def load_tokens_and_qids(file_path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    d = np.load(file_path)
    return d["tokens"], d["qids"]


class AliasTableLoader:
    def __init__(
        self, mewsli_root_path: Path, damuel_root_path: Path, lowercase: bool = False
    ):
        self.mewsli_root_path = mewsli_root_path
        self.damuel_root_path = damuel_root_path
        self.lowercase = lowercase

    @remap_qids_decorator(qids_index=1, json_path=gin.REQUIRED)
    def load_mewsli(self, lang: str) -> tuple[list[str], list[int]]:
        df = pd.read_csv(self._construct_mewsli_path(lang), sep="\t")
        if self.lowercase:
            df["mention"] = df["mention"].str.lower()
        return df["mention"].tolist(), df["qid"].apply(lambda x: int(x[1:])).tolist()

    def _construct_mewsli_path(self, lang: str) -> Path:
        return self.mewsli_root_path / lang / "mentions.tsv"
