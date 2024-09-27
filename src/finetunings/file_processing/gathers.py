import os
import pickle
import shutil
from cmath import inf
from pathlib import Path

import numpy as np

from utils.argument_wrappers import ensure_datatypes


def _wanted_hash(hash_str, m, r):
    hash_int = int(hash_str, 16)
    return hash_int % m == r


def _count_tokens(fp: Path):
    return len(np.load(fp)["tokens"])


def _get_hash_from_fn(fn):
    return fn.split("_")[-1].split(".")[0]


def _wanted_fn(fn: str, m: int, r: int):
    return fn.endswith("npz") and _wanted_hash(_get_hash_from_fn(fn), m, r)


@ensure_datatypes([Path, Path, int, int], {})
def move_tokens(source, dest, m=1, r=0, max_to_copy: int = inf):
    already_copied = 0
    for fn in sorted(os.listdir(source)):
        if not _wanted_fn(fn, m, r):
            continue
        is_enough_copied = already_copied >= max_to_copy
        if is_enough_copied:
            break
        tokens_cnt = _count_tokens(source / fn)
        already_copied += tokens_cnt
        shutil.copy(os.path.join(source, fn), dest)


@ensure_datatypes([Path, str, str], {})
def rename(dest, orig, new):
    for fn in os.listdir(dest):
        if orig in fn:
            new_fn = fn.replace(orig, new)
            os.rename(os.path.join(dest, fn), os.path.join(dest, new_fn))


@ensure_datatypes([Path], {})
def remove_duplicates(source):
    seen = set()
    for fn in os.listdir(source):
        if fn.endswith("xz"):
            continue
        with open(os.path.join(source, fn), "rb") as f:
            data = pickle.load(f)
        data = set(data)
        print(f"Before: {len(data)}")
        seen = seen.union(data)
        print(f"After: {len(seen)}")
    with open(os.path.join(source, "mentions_all"), "wb") as f:
        pickle.dump(list(seen), f)
