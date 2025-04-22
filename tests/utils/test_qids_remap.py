import json
from unittest.mock import patch, Mock

import numpy as np
import pytest

from utils.qids_remap import load_qids_remap, qids_remap, remap_qids_decorator
import utils.qids_remap as qr  # to reset the internal cache


def test_load_qids_remap(tmp_path):
    # Create a temporary JSON file with QID mappings
    test_qid_map = {"Q1": "Q10", "Q2": "Q20", "Q3": "Q30"}
    test_file = tmp_path / "qid_redirects.json"
    with open(test_file, "w") as f:
        json.dump(test_qid_map, f)

    # Load and check the data
    loaded_qid_map = load_qids_remap(test_file)

    assert loaded_qid_map == {1: 10, 2: 20, 3: 30}


def test_qids_remap():
    mock_qid_map = {1: 10, 2: 20, 3: 30}
    with patch("utils.qids_remap.load_qids_remap", return_value=mock_qid_map):
        qr._qids_lookup = None  # clear any previously built table

        input_qids = np.array([1, 2, 3, 4, 5])
        expected_output = np.array([10, 20, 30, 4, 5])

        remapped_qids = qids_remap(input_qids, "dummy_path")

        assert np.array_equal(remapped_qids, expected_output)


@pytest.mark.parametrize(
    "dtype", [np.int16, np.int32, np.int64, np.uint16, np.uint32, np.uint64]
)
def test_qids_remap_preserve_dtype(dtype):
    mock_qid_map = {1: 10, 2: 20, 3: 30}
    with patch("utils.qids_remap.load_qids_remap", return_value=mock_qid_map):
        qr._qids_lookup = None

        input_qids = np.array([1, 2, 3, 4, 5], dtype=dtype)
        remapped_qids = qids_remap(input_qids, "dummy_path")

        assert remapped_qids.dtype == dtype


def test_qids_remap_decorator():
    mock_qid_map = {1: 10, 2: 20, 3: 30}
    with patch("utils.qids_remap.load_qids_remap", return_value=mock_qid_map):
        qr._qids_lookup = None

        @remap_qids_decorator(qids_index=None, json_path="dummy_path")
        def test_func(arg1: str, qids: np.ndarray) -> np.ndarray:
            return qids

        input_qids = np.array([1, 2, 3, 4, 5])
        expected_output = np.array([10, 20, 30, 4, 5])

        remapped_qids = test_func("some_arg", input_qids)

        assert np.array_equal(remapped_qids, expected_output)


def test_qids_remap_decorator_with_index():
    mock_qid_map = {1: 10, 2: 20, 3: 30}
    with patch("utils.qids_remap.load_qids_remap", return_value=mock_qid_map):
        qr._qids_lookup = None

        @remap_qids_decorator(qids_index=1, json_path="dummy_path")
        def test_func(arg1: str, qids: np.ndarray):
            return None, qids

        input_qids = np.array([1, 2, 3, 4, 5])
        expected_output = np.array([10, 20, 30, 4, 5])

        _, remapped_qids = test_func("some_arg", input_qids)

        assert np.array_equal(remapped_qids, expected_output)


def test_load_only_once():
    qr._qids_lookup = None
    m = Mock(return_value={1: 10})
    with patch("utils.qids_remap.load_qids_remap", m):
        a = qids_remap(np.array([1]), "p")
        b = qids_remap(np.array([1]), "p")
    assert m.call_count == 1


def test_large_qid_fallback():
    qr._qids_lookup = None
    with patch("utils.qids_remap.load_qids_remap", return_value={1: 10}):
        inp = np.array([1, 9999], dtype=np.int32)
        with pytest.raises(IndexError):
            out = qids_remap(inp, "dummy")
