import hashlib
import lzma
from pathlib import Path
import pickle
import sys
import numpy as np

from fire import Fire


sys.path.append("/home/farhand/bc/src")

from utils.argument_wrappers import ensure_datatypes

_LEALLA_PADDING_TOKEN = 0


def hash_str(s):
    return int(hashlib.sha1(s.encode()).hexdigest(), 16) % 10**8


def lealla_padding_policy(mention_qid_pair):
    tokens = mention_qid_pair.tokenization_output["input_ids"][0]
    return -(tokens == _LEALLA_PADDING_TOKEN).sum()


def process_data(data, path):
    res = np.zeros((len(data), 4), int)
    path_hash = hash_str(str(path))
    for i, x in enumerate(data):
        res[i] = [x.qid, lealla_padding_policy(x), i, path_hash]
    return res


@ensure_datatypes([Path, Path], {})
def solve(source_dir, dest):
    print(f"Processing {source_dir}")
    with lzma.open(source_dir, "rb") as f:
        data = pickle.load(f)
    data = process_data(data, source_dir)
    out_path = Path(dest) / source_dir.name.replace(".xz", ".npy")
    np.save(out_path, data)


if __name__ == "__main__":
    Fire(solve)
