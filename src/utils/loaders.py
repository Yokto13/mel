import functools
import os
from pathlib import Path

import gin
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix, csr_matrix
from tqdm import tqdm

# from tokenization.pipeline import DamuelAliasTablePipeline
from tokenization.runner import run_alias_table_damuel
from utils.qid_filter import qid_filter
from utils.qids_remap import remap_qids_decorator

current_file_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))
config_path = os.path.join(project_root, "configs", "general.gin")
gin.parse_config_file(config_path)
# config_path = os.path.join(project_root, "configs", "multilingual_dataset.gin")
# gin.parse_config_file(config_path)


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
@qid_filter(qids_index=1)
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
@qid_filter(qids_index=1)
@remap_qids_decorator(qids_index=1, json_path=gin.REQUIRED)
def load_tokens_qids(file_path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Loads 'tokens' and 'qids' arrays from a NumPy .npz file.

    Args:
        file_path (str | Path): Path to the .npz file containing 'tokens' and 'qids' arrays.

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing the 'tokens' and 'qids' arrays.
    """
    if type(file_path) == str:
        file_path = Path(file_path)
    d = np.load(file_path)
    return d["tokens"], d["qids"]


@qid_filter(qids_index=None)
@remap_qids_decorator(qids_index=None, json_path=gin.REQUIRED)
def load_qids(file_path: str | Path) -> np.ndarray:
    if type(file_path) == str:
        file_path = Path(file_path)
    d = np.load(file_path)
    return d["qids"]


@qid_filter(qids_index=None)
@remap_qids_decorator(qids_index=None, json_path=gin.REQUIRED)
def load_qids_npy(file_path: str | Path) -> np.ndarray:
    if type(file_path) == str:
        file_path = Path(file_path)
    return np.load(file_path)


@_sort_by_output(1)
@qid_filter(qids_index=1)
@remap_qids_decorator(qids_index=1, json_path=gin.REQUIRED)
def load_tokens_qids_from_dir(
    dir_path: str | Path, verbose=False, max_items_to_load: int | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """
    Loads mention tokens and query IDs from all .npz files in a given directory.

    Args:
        dir_path (str | Path): Path to the directory containing .npz files.
        verbose (bool): If True, displays a progress bar while loading files.

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing two numpy arrays:
            - tokens: Array of mention tokens loaded from the files.
            - qids: Array of query IDs loaded from the files.
    """
    if type(dir_path) == str:
        dir_path = Path(dir_path)
    tokens, qids = [], []
    iterator = dir_path.iterdir()
    if verbose:
        total = sum(1 for itm in dir_path.iterdir() if itm.is_file() and itm.suffix == ".npz")
        iterator = tqdm(
            dir_path.iterdir(), desc=f"Loading tokens and qids from {dir_path}", total=total
        )
    for file in iterator:
        if file.is_file() and file.suffix == ".npz":
            d = np.load(file)
            tokens.extend(d["tokens"])
            qids.extend(d["qids"])
        if max_items_to_load is not None and len(tokens) >= max_items_to_load:
            break
    return np.array(tokens), np.array(qids)


@qid_filter(qids_index=1)
@remap_qids_decorator(qids_index=1, json_path=gin.REQUIRED)
def load_tokens_and_qids(file_path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    d = np.load(file_path)
    return d["tokens"], d["qids"]


def map_qids_to_token_matrix(
    dir_path: str | Path, verbose: bool = False, max_items_to_load: int | None = None
) -> csr_matrix:
    """Builds a memory-efficient sparse matrix mapping qids to their token vectors.

    Args:
        dir_path (str | Path): Directory containing data files with 'tokens' and 'qids'.
        verbose (bool): Forwarded to `load_tokens_qids_from_dir` to toggle progress output.
        max_items_to_load (int | None): Optional cap on the number of token rows to read.

    Returns:
        scipy.sparse.csr_matrix: A CSR matrix where a row index corresponds to a qid
                                 and the row's data is the token vector. Use
                                 `matrix[qid]` to retrieve a vector.
    """
    tokens, qids = load_tokens_qids_from_dir(
        dir_path=dir_path, verbose=verbose, max_items_to_load=max_items_to_load
    )

    num_items, vector_len = tokens.shape

    assert num_items == qids.shape[0], "Mismatch between number of token rows and qids"

    row_indices = np.repeat(qids, vector_len)
    col_indices = np.tile(np.arange(vector_len), num_items)
    data = tokens.flatten()

    shape = (qids.max() + 1, vector_len)

    coo = coo_matrix((data, (row_indices, col_indices)), shape=shape, dtype=tokens.dtype)
    return coo.tocsr()


class AliasTableLoader:
    """
    This class provides methods to load and process alias tables from two different sources:
        - MEWSLI alias tables, stored as tab-separated files.
        - DAMUEL alias tables, processed via a dedicated pipeline.

    Attributes:
            mewsli_root_path (Path): Base directory containing MEWSLI alias table files.
            damuel_root_path (Path): Base directory where directories for DAMUEL alias tables reside.
            lowercase (bool): Flag to indicate whether mentions should be converted to lowercase.

    TODO: Move as much logic as possible to the pipeline. Probably just get rid of this class.
    """

    def __init__(self, mewsli_root_path: Path, damuel_root_path: Path, lowercase: bool = False):
        self.mewsli_root_path = mewsli_root_path
        self.damuel_root_path = damuel_root_path
        self.lowercase = lowercase

    @remap_qids_decorator(qids_index=1, json_path=gin.REQUIRED)
    def load_mewsli(self, lang: str) -> tuple[list[str], np.ndarray]:
        df = pd.read_csv(self._construct_mewsli_path(lang), sep="\t")
        if self.lowercase:
            df["mention"] = df["mention"].str.lower()
        return df["mention"].tolist(), df["qid"].apply(lambda x: int(x[1:])).to_numpy()

    @qid_filter(qids_index=1)
    @remap_qids_decorator(qids_index=1, json_path=gin.REQUIRED)
    def load_damuel(self, lang) -> tuple[list[str], np.ndarray]:
        data = run_alias_table_damuel(self._construct_damuel_path(lang))
        textual = np.concatenate([[x[0] for x in d] for d in data])
        qids = np.concatenate([[x[1] for x in d] for d in data])
        if self.lowercase:
            textual = [t.lower() for t in textual]
        print(qids)
        return textual, qids

    def _construct_mewsli_path(self, lang: str) -> str:
        return (self.mewsli_root_path / lang / "mentions.tsv").as_posix()

    def _construct_damuel_path(self, lang: str) -> str:
        for subdir in self.damuel_root_path.iterdir():
            if subdir.is_dir() and subdir.name.endswith(lang):
                print(f"Found directory: {subdir}")
                return subdir.as_posix()
        raise FileNotFoundError(
            f"No directory ending with '{lang}' found in {self.damuel_root_path}"
        )
