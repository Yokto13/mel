import os
from collections import Counter
import numpy as np
from tqdm import tqdm
import fire

"""
Given a path to a directory of multilingual dataset, calculates qid occurrence frequencies.
Usage: python qid_occurences.py <DATASET_PATH>
Assumes that the directory contains files with suffix <lang>_<number>.npz
"""

import concurrent.futures


def _file_qid_counts(path):
    data = np.load(path)
    qids = data["qids"]
    data.close()
    # Count occurrences of each qid in this file
    return Counter(qids.flatten())


def calculate_qid_statistics(directory_path=".", workers: int | None = None):
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

    total_counter = Counter()

    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_file_qid_counts, p): p for p in npz_files}
        for fut in tqdm(
            concurrent.futures.as_completed(futures),
            total=len(futures),
            desc="Processing files",
        ):
            try:
                file_counter = fut.result()
                total_counter.update(file_counter)
            except Exception as e:
                print(f"Error {futures[fut]}: {e}")
                continue

    if not total_counter:
        print("No qids found.")
        return

    # Get occurrence counts for statistical analysis
    occurrence_counts = list(total_counter.values())

    # Calculate statistics
    average_count = np.mean(occurrence_counts)
    median_count = np.median(occurrence_counts)
    q1_count = np.percentile(occurrence_counts, 25)
    q3_count = np.percentile(occurrence_counts, 75)

    # Calculate all deciles (10th, 20th, ..., 90th percentiles)
    deciles = [np.percentile(occurrence_counts, p) for p in range(10, 100, 10)]

    # Get top 10 most occurring and top 10 least occurring
    most_common = total_counter.most_common(10)
    least_common = total_counter.most_common()[-10:]

    print("\n" + "=" * 50)
    print("QID OCCURRENCE STATISTICS")
    print("=" * 50)
    print(f"Total unique qids: {len(total_counter):,}")
    print(f"Total qid occurrences: {sum(total_counter.values()):,}")
    print(f"Average occurrence count: {average_count:.2f}")
    print(f"Median occurrence count: {median_count:.2f}")
    print(f"Q1 (25th percentile): {q1_count:.2f}")
    print(f"Q3 (75th percentile): {q3_count:.2f}")

    print("\nDECILES:")
    for i, decile in enumerate(deciles, 1):
        print(f"{i*10:2d}th percentile: {decile:.2f}")

    print("\nTOP 10 MOST OCCURRING QIDS:")
    for i, (qid, count) in enumerate(most_common, 1):
        print(f"{i:2d}. QID {qid}: {count:,} occurrences")

    print("\nTOP 10 LEAST OCCURRING QIDS:")
    for i, (qid, count) in enumerate(least_common, 1):
        print(f"{i:2d}. QID {qid}: {count:,} occurrences")
    print("=" * 50)


if __name__ == "__main__":
    fire.Fire(calculate_qid_statistics)
