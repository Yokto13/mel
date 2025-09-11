"""
Given a path to a directory of multilingual dataset, calculates how many entities are there per language.
Usage: python lang_statistics.py <DATASET_PATH>

Assumes that the directory contains files with suffix <lang>_<number>.npz
"""

import os
from collections import defaultdict

import fire
import numpy as np
from tqdm import tqdm


def count_qids_by_language(directory_path="."):
    """
    Count qids by language from .npz files in the given directory.

    Args:
        directory_path (str): Path to the directory containing .npz files
    """
    # Counter mapping language code to number of qids
    lang_counter = defaultdict(int)

    # Get list of .npz files
    npz_files = [f for f in os.listdir(directory_path) if f.endswith(".npz")]

    if not npz_files:
        print("No .npz files found in the directory.")
        return

    # Iterate through all files with progress bar
    for filename in tqdm(npz_files, desc="Processing files"):
        # Extract language code from filename (format: <lang>_<number>.npz)
        lang_code = filename.split("_")[-2]

        try:
            # Load the .npz file
            filepath = os.path.join(directory_path, filename)
            data = np.load(filepath)

            # Get qids and count them
            if "qids" in data:
                qids = data["qids"]
                qid_count = len(qids)
                lang_counter[lang_code] += qid_count
            else:
                print(f"Warning: 'qids' key not found in {filename}")

            # Close the file
            data.close()

        except Exception as e:
            print(f"Error processing {filename}: {e}")

    # Sort languages by count (most to least)
    sorted_langs = sorted(lang_counter.items(), key=lambda x: x[1], reverse=True)

    # Display results
    print("\n" + "=" * 50)
    print("LANGUAGE STATISTICS (Most to Least Used)")
    print("=" * 50)

    if not sorted_langs:
        print("No .npz files found or no qids data available.")
        return

    total_qids = sum(lang_counter.values())

    for rank, (lang, count) in enumerate(sorted_langs, 1):
        percentage = (count / total_qids) * 100 if total_qids > 0 else 0
        print(f"{rank:2d}. {lang:10s}: {count:8,} qids ({percentage:5.1f}%)")

    print("-" * 50)
    print(f"Total languages: {len(sorted_langs)}")
    print(f"Total qids: {total_qids:,}")


if __name__ == "__main__":
    fire.Fire(count_qids_by_language)
