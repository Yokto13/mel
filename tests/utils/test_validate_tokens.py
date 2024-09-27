import os

import numpy as np
import pytest
from utils.validate_tokens import validate_tokens


def _create_test_directories(structure, current_path):
    for name, content in structure.items():
        file_path = current_path / name
        if name.endswith(".npz"):
            np.savez(file_path, tokens=content["tokens"], qids=content["qids"])
            return
        file_path.mkdir(parents=True, exist_ok=True)
        _create_test_directories(content, file_path)


def test_validate_tokens_equal_directories(tmp_path):
    structure = {
        "dir1": {
            "subdir1": {
                "file1.npz": {
                    "tokens": np.array([[1, 2, 3], [4, 5, 6]]),
                    "qids": np.array([1, 2]),
                },
                "file2.npz": {
                    "tokens": np.array([[7, 8, 9], [10, 11, 12]]),
                    "qids": np.array([3, 4]),
                },
            },
            "subdir2": {
                "file3.npz": {
                    "tokens": np.array([[13, 14, 15], [16, 17, 18]]),
                    "qids": np.array([5, 6]),
                },
            },
        },
        "dir2": {
            "subdir1": {
                "file1.npz": {
                    "tokens": np.array([[1, 2, 3], [4, 5, 6]]),
                    "qids": np.array([1, 2]),
                },
                "file2.npz": {
                    "tokens": np.array([[7, 8, 9], [10, 11, 12]]),
                    "qids": np.array([3, 4]),
                },
            },
            "subdir2": {
                "file3.npz": {
                    "tokens": np.array([[13, 14, 15], [16, 17, 18]]),
                    "qids": np.array([5, 6]),
                },
            },
        },
    }
    _create_test_directories(structure, tmp_path)
    dir1 = tmp_path / "dir1"
    dir2 = tmp_path / "dir2"
    assert validate_tokens(str(dir1), str(dir2))


def test_validate_tokens_unequal_directories(tmp_path):
    structure = {
        "dir1": {
            "subdir1": {
                "file1.npz": {
                    "tokens": np.array([[1, 2, 3], [4, 5, 6]]),
                    "qids": np.array([1, 2]),
                },
                "file2.npz": {
                    "tokens": np.array([[7, 8, 9], [10, 11, 12]]),
                    "qids": np.array([3, 4]),
                },
            },
            "subdir2": {
                "file3.npz": {
                    "tokens": np.array([[13, 14, 15], [16, 17, 18]]),
                    "qids": np.array([5, 6]),
                },
            },
        },
        "dir2": {
            "subdir1": {
                "file1.npz": {
                    "tokens": np.array([[1, 2, 3], [4, 5, 6]]),
                    "qids": np.array([1, 2]),
                },
                "file2.npz": {
                    "tokens": np.array([[7, 8, 9], [10, 11, 12]]),
                    "qids": np.array([3, 4]),
                },
            },
            "subdir2": {
                "file3.npz": {
                    "tokens": np.array([[13, 14, 15], [16, 17, 19]]),
                    "qids": np.array([5, 6]),
                },
            },
        },
    }
    _create_test_directories(structure, tmp_path)
    dir1 = tmp_path / "dir1"
    dir2 = tmp_path / "dir2"
    assert not validate_tokens(str(dir1), str(dir2))


def test_validate_tokens_non_directory_arguments():
    with pytest.raises(ValueError, match="Both arguments must be directories."):
        validate_tokens("not_a_directory", "another_not_a_directory")


def test_validate_tokens_different_subdirectories(tmp_path):
    structure = {
        "dir1": {
            "subdir1": {
                "file1.npz": {
                    "tokens": np.array([[1, 2, 3], [4, 5, 6]]),
                    "qids": np.array([1, 2]),
                },
                "file2.npz": {
                    "tokens": np.array([[7, 8, 9], [10, 11, 12]]),
                    "qids": np.array([3, 4]),
                },
            },
        },
        "dir2": {
            "subdir2": {
                "file1.npz": {
                    "tokens": np.array([[1, 2, 3], [4, 5, 6]]),
                    "qids": np.array([1, 2]),
                },
                "file2.npz": {
                    "tokens": np.array([[7, 8, 9], [10, 11, 12]]),
                    "qids": np.array([3, 4]),
                },
            },
        },
    }
    _create_test_directories(structure, tmp_path)
    dir1 = tmp_path / "dir1"
    dir2 = tmp_path / "dir2"
    assert not validate_tokens(str(dir1), str(dir2))
