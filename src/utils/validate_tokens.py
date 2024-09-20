""" The way tokens are generated changed throughout ages.

    This script provides method that checks recursivelly that takes two directories and recursively checks
    that contents of subdirectories are equal.
    The leaf directories are expected to contain npz files.
    These files are expected to contain tokens and qids.
    To leaf directories are equal if data from all files are equal.
    To check that we sort data by qid.
    To inner node directories are equal if all their children are equal.
"""

import os
import numpy as np


def validate_tokens(dir1, dir2):
    if not os.path.isdir(dir1) or not os.path.isdir(dir2):
        raise ValueError("Both arguments must be directories.")

    comparison_results = _compare_directories(dir1, dir2)
    if comparison_results:
        print("Directories are equal")
    else:
        print("Directories are not equal")
    return comparison_results


def _compare_directories(dir1, dir2):
    files1 = set(os.listdir(dir1))
    files2 = set(os.listdir(dir2))

    if _are_leaf_directories(files1, files2):
        print("comparing leaf directories")
        print(dir1)
        print(dir2)
        print(files1)
        print(files2)
        return _compare_leaf_directories(dir1, dir2)

    if files1 != files2:
        return False

    same = True
    for file in files1:
        path1 = os.path.join(dir1, file)
        path2 = os.path.join(dir2, file)

        if os.path.isdir(path1) and os.path.isdir(path2):
            same &= _compare_directories(path1, path2)

        else:
            same = False

    if not same:
        print("Failing in the following directories:")
        print(dir1)
        print(dir2)
        print(f"Directories are not equal")
    return same


def _are_leaf_directories(files1, files2):
    return all(file.endswith(".npz") for file in files1) and all(
        file.endswith(".npz") for file in files2
    )


def _compare_leaf_directories(dir1, dir2):
    npz_files1 = [file for file in os.listdir(dir1) if file.endswith(".npz")]
    npz_files2 = [file for file in os.listdir(dir2) if file.endswith(".npz")]

    tokens1 = []
    qids1 = []
    for file in npz_files1:
        data = np.load(os.path.join(dir1, file))
        tokens1.append(data["tokens"])
        qids1.append(data["qids"])

    tokens2 = []
    qids2 = []
    for file in npz_files2:
        data = np.load(os.path.join(dir2, file))
        tokens2.append(data["tokens"])
        qids2.append(data["qids"])

    tokens1 = np.concatenate(tokens1)
    qids1 = np.concatenate(qids1)
    tokens2 = np.concatenate(tokens2)
    qids2 = np.concatenate(qids2)

    if len(tokens1) != len(tokens2) or len(qids1) != len(qids2):
        print("Failing in the following directories:")
        print(dir1)
        print(dir2)
        print(
            f"Lengths of tokens and qids are not equal: {len(tokens1)} != {len(tokens2)} or {len(qids1)} != {len(qids2)}"
        )
        return False

    sorted_indices1 = np.argsort(qids1)
    sorted_indices2 = np.argsort(qids2)

    sorted_tokens1 = tokens1[sorted_indices1]
    sorted_tokens2 = tokens2[sorted_indices2]
    sorted_qids1 = qids1[sorted_indices1]
    sorted_qids2 = qids2[sorted_indices2]

    are_equal = np.array_equal(sorted_tokens1, sorted_tokens2) and np.array_equal(
        sorted_qids1, sorted_qids2
    )
    if not are_equal:
        print("Failing in the following directories:")
        print(dir1)
        print(dir2)
        print(f"Tokens and qids are not equal")
    return are_equal


def _compare_files(file1, file2):
    with open(file1, "rb") as f1, open(file2, "rb") as f2:
        return f1.read() == f2.read()
