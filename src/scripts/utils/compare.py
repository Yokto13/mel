#!/usr/bin/env python3
"""
compare_qids.py  – Compare “qids” sets contained in .npz files
Usage:  python compare_qids.py <DIR_A> <DIR_B>
"""

import sys
from pathlib import Path

import numpy as np


def collect_qids(dir_path: Path) -> set:
    """Return the union of all qids found in .npz files under dir_path."""
    qids: set = set()
    for file in dir_path.glob("*.npz"):
        with np.load(file, allow_pickle=True) as data:
            if "qids" in data:
                qids.update(data["qids"].tolist())
    return qids


def main(dir_a: Path, dir_b: Path) -> None:
    q_a = collect_qids(dir_a)
    q_b = collect_qids(dir_b)

    only_a = q_a - q_b
    only_b = q_b - q_a
    common = q_a & q_b

    print(f"Total unique qids in {dir_a}: {len(q_a)}")
    print(f"Total unique qids in {dir_b}: {len(q_b)}")

    print(f"Unique to {dir_a} ({len(only_a)}):")
    print(f"Unique to {dir_b} ({len(only_b)}):")
    print(f"Common to both ({len(common)})")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("Usage: python compare_qids.py <DIR_A> <DIR_B>")
    main(Path(sys.argv[1]), Path(sys.argv[2]))
