import functools
import json
import lzma
import os
from pathlib import Path

import numpy as np
import pandas as pd
import gin

from utils.qids_remap import remap_qids_decorator


current_file_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))
config_path = os.path.join(project_root, "configs", "general.gin")
gin.parse_config_file(config_path)


@remap_qids_decorator(qids_index=1, json_path=gin.REQUIRED)
def load_mewsli(tsv_path: Path, lowercase=False) -> tuple[list[str], list[str]]:
    df = pd.read_csv(tsv_path, sep="\t")
    if lowercase:
        df["mention"] = df["mention"].str.lower()
    # if "fa" in tsv_path.parts:
    #     normalizer = Normalizer()
    #     print("normalizing farsi")
    #     df["mention"] = df["mention"].apply(normalizer.normalize)

    return df["mention"].tolist(), df["qid"].apply(lambda x: int(x[1:])).tolist()


@remap_qids_decorator(qids_index=1, json_path=gin.REQUIRED)
def load_damuel(
    dir_path: Path, only_wiki: bool, use_xz=False, lowercase=False
) -> tuple[list[str], list[str]]:
    def process_line(line):
        loaded = json.loads(line)
        if "wiki" not in loaded:
            return
        wiki = loaded["wiki"]
        tokens = wiki["tokens"]
        text = wiki["text"]
        if lowercase:
            text = text.lower()
        # if "damuel_1.0_fa" in dir_path.parts:
        #     normalizer = Normalizer()
        #     text = normalizer.normalize(text)
        for l in wiki["links"]:
            if "qid" not in l:
                continue
            if only_wiki and l["origin"] != "wiki":
                continue
            start = l["start"]
            end = l["end"] - 1
            try:
                mention_slice = slice(tokens[start]["start"], tokens[end]["end"])
            except IndexError:
                print(start, end, len(tokens))
            mention_names.append(text[mention_slice])
            qids.append(int(l["qid"][1:]))

    # if "damuel_1.0_fa" in dir_path.parts:
    #     print("normalizing farsi")

    mention_names, qids = [], []
    for fn in dir_path.iterdir():
        print("processing", fn)
        if use_xz:
            with lzma.open(fn, "r") as f:
                for line in f:
                    process_line(line)
        else:
            for line in fn.open():
                process_line(line)
        print("processed", fn)
    return mention_names, qids


@remap_qids_decorator(qids_index=1, json_path=gin.REQUIRED)
def load_mewsli_from_files(tsv_path: Path) -> tuple[list[str], list[str]]:
    df = pd.read_csv(tsv_path, sep="\t")
    mentions, qids = [], []
    for row in df.itertuples():
        with open(tsv_path.parent / "text" / row.docid, "r") as f:
            text = f.read()
            mentions.append(text[row.position : row.position + row.length])
            qids.append(int(row.qid[1:]))
    return mentions, qids


@remap_qids_decorator(qids_index=1, json_path=gin.REQUIRED)
def load_damuel_context(
    dir_path: Path, only_wiki: bool, context_char_size
) -> tuple[list[str], list[str]]:
    contexts, qids = [], []
    for fn in dir_path.iterdir():
        for line in fn.open():
            loaded = json.loads(line)
            if "wiki" not in loaded:
                continue
            wiki = loaded["wiki"]
            tokens = wiki["tokens"]
            text = wiki["text"]
            for l in wiki["links"]:
                if "qid" not in l:
                    continue
                if only_wiki and l["origin"] != "wiki":
                    continue
                start = l["start"]
                end = l["end"] - 1
                # assert start >= 0
                # assert end < len(tokens)
                # print(start, end, len(tokens))
                char_start = tokens[start]["start"]
                char_end = tokens[end]["end"]
                char_start = max(0, char_start - context_char_size)
                char_end = min(len(text), char_end + context_char_size)
                contexts.append(text[char_start:char_end])
                qids.append(int(l["qid"][1:]))
        print("processed", fn)
    return contexts, qids


@remap_qids_decorator(qids_index=1, json_path=gin.REQUIRED)
def load_mewsli_context_from_files(
    tsv_path: Path, context_char_size
) -> tuple[list[str], list[str]]:
    df = pd.read_csv(tsv_path, sep="\t")
    contexts, qids = [], []
    for row in df.itertuples():
        with open(tsv_path.parent / "text" / row.docid, "r") as f:
            text = f.read()
            char_start = max(0, row.position - context_char_size)
            char_end = min(len(text), row.position + row.length + context_char_size)
            contexts.append(text[char_start:char_end])
            qids.append(int(row.qid[1:]))
    return contexts, qids


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
def load_embs_qids_tokens(dir_path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Loads embeddings, qids, and tokens from the directory.

    This should be the preferable than directly loading them to ensure that the logic of loading is just in one place.

    Args:
        dir_path (str | Path)

    Returns:
        tuple[np.ndarray, np.ndarray]: embs, qids, tokens
    """
    if type(dir_path) == str:
        dir_path = Path(dir_path)
    d = np.load(dir_path / "embs_qids_tokens.npz")
    return d["embs"], d["qids"], d["tokens"]


# @_sort_by_output(1)
@remap_qids_decorator(qids_index=1, json_path=gin.REQUIRED)
def load_mentions(file_path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    if type(file_path) == str:
        file_path = Path(file_path)
    d = np.load(file_path)
    return d["tokens"], d["qids"]


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
