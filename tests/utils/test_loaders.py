import pytest
import numpy as np
from pathlib import Path
import tempfile

from utils.loaders import (
    load_mentions,
)


def test_load_mentions_with_path_object():
    with tempfile.TemporaryDirectory() as temp_dir:
        file_path = Path(temp_dir) / "mentions_2.npz"

        test_tokens = np.array([[1, 2, 3], [4, 5, 6]])
        test_qids = np.array([100, 200])

        np.savez_compressed(file_path, tokens=test_tokens, qids=test_qids)

        loaded_tokens, loaded_qids = load_mentions(file_path)

        assert np.array_equal(loaded_tokens, test_tokens)
        assert np.array_equal(loaded_qids, test_qids)


def test_load_mentions_with_string_path():
    with tempfile.TemporaryDirectory() as temp_dir:
        file_path = str(Path(temp_dir) / "mentions_1.npz")

        test_tokens = np.array([[10, 20], [30, 40], [50, 60]])
        test_qids = np.array([1000, 2000, 3000])

        np.savez_compressed(file_path, tokens=test_tokens, qids=test_qids)

        loaded_tokens, loaded_qids = load_mentions(file_path)

        assert np.array_equal(loaded_tokens, test_tokens)
        assert np.array_equal(loaded_qids, test_qids)

        assert isinstance(loaded_tokens, np.ndarray)
        assert isinstance(loaded_qids, np.ndarray)
