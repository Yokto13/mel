import json
import numpy as np
from unittest.mock import patch
from utils.qids_remap import load_qids_remap, qids_remap

import pytest


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
    # Mock the load_qids_remap function
    mock_qid_map = {1: 10, 2: 20, 3: 30}
    with patch("utils.qids_remap.load_qids_remap", return_value=mock_qid_map):
        # Test qids_remap function
        input_qids = np.array([1, 2, 3, 4, 5])
        expected_output = np.array([10, 20, 30, 4, 5])

        remapped_qids = qids_remap(input_qids, "dummy_path")

        assert np.array_equal(remapped_qids, expected_output)


@pytest.mark.parametrize(
    "dtype", [np.int16, np.int32, np.int64, np.uint16, np.uint32, np.uint64]
)
def test_qids_remap_preserve_dtype(dtype):
    # Mock the load_qids_remap function
    mock_qid_map = {1: 10, 2: 20, 3: 30}
    with patch("utils.qids_remap.load_qids_remap", return_value=mock_qid_map):
        # Test qids_remap function
        input_qids = np.array([1, 2, 3, 4, 5], dtype=dtype)

        remapped_qids = qids_remap(input_qids, "dummy_path")

        assert remapped_qids.dtype == dtype
