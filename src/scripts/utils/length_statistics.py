"""
Given a path to a directory of multilingual dataset, calculates average and std of token lengths.
Usage: python length_statistics.py <DATASET_PATH>
Assumes that the directory contains files with suffix <lang>_<number>.npz
"""

import concurrent.futures
import math
import os

import fire
import numpy as np
from tqdm import tqdm


def _file_stats(path):
    data = np.load(path)
    tokens = data["tokens"]
    lengths = np.count_nonzero(tokens, axis=1)
    data.close()
    count = lengths.size
    sum_len = lengths.sum(dtype=np.int64)
    sum_sq = np.square(lengths, dtype=np.int64).sum(dtype=np.int64)
    return count, sum_len, sum_sq


def calculate_length_statistics(directory_path=".", workers: int | None = None):
    npz_files = [
        os.path.join(directory_path, f)
        for f in os.listdir(directory_path)
        if f.endswith(".npz")
    ]
    if not npz_files:
        print("No .npz files found.")
        return
    if workers is None:
        workers = min(32, (os.cpu_count() or 1))

    total_count = 0
    total_sum = 0
    total_sum_sq = 0

    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_file_stats, p): p for p in npz_files}
        for fut in tqdm(
            concurrent.futures.as_completed(futures),
            total=len(futures),
            desc="Processing files",
        ):
            try:
                count, s, ss = fut.result()
            except Exception as e:
                print(f"Error {futures[fut]}: {e}")
                continue
            total_count += count
            total_sum += s
            total_sum_sq += ss

    if not total_count:
        print("No sequences found.")
        return

    mean = total_sum / total_count
    variance = (total_sum_sq / total_count) - mean * mean
    std = math.sqrt(max(variance, 0.0))

    print("\n" + "=" * 50)
    print("TOKEN LENGTH STATISTICS")
    print("=" * 50)
    print(f"Total sequences processed: {total_count:,}")
    print(f"Average length: {mean:.2f}")
    print(f"Standard deviation: {std:.2f}")
    print(f"Minimum length: 0 (assumed)")
    print("=" * 50)


if __name__ == "__main__":
    fire.Fire(calculate_length_statistics)
