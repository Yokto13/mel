import os
import shutil
import pickle
from pathlib import Path

from finetunings.file_processing.compressions import decompress_files_in_directory
from utils.argument_wrappers import ensure_datatypes


def _wanted_hash(hash_str, m, r):
    hash_int = int(hash_str, 16)
    return hash_int % m == r


def _copy_all_tokens(source, dest, m=1, r=0):
    for fn in sorted(os.listdir(source)):
        print(fn)
        if not fn.endswith("npz"):
            continue
        hash_str = fn.split("_")[-1].split(".")[0]
        if not _wanted_hash(hash_str, m, r):
            continue
        shutil.copy(os.path.join(source, fn), dest)


@ensure_datatypes([Path, Path, int, int, bool], {})
def move_tokens(source, dest, m=1, r=0):
    _copy_all_tokens(source, dest, m, r)


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
