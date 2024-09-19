import json
import pytest
import numpy as np
from unittest.mock import patch

from finetunings.evaluation.find_recall import (
    load_embs_and_qids_with_normalization,
    get_scann_searcher,
    get_brute_force_searcher,
    get_faiss_searcher,
    load_qids_remap,
    qids_remap,
)
from models.searchers.scann_searcher import ScaNNSearcher
from models.searchers.brute_force_searcher import BruteForceSearcher
from models.searchers.faiss_searcher import FaissSearcher


@pytest.fixture
def dummy_data():
    embs = np.random.rand(100000, 100).astype(np.float32)
    qids = np.arange(100000, dtype=np.int64)
    return embs, qids


def test_load_embs_and_qids_with_normalization(tmp_path):
    # Create a temporary npz file
    test_embs = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    test_qids = np.array([1, 2], dtype=np.int64)
    test_file = tmp_path / "embs_qids.npz"
    np.savez(test_file, embs=test_embs, qids=test_qids)

    # Load and check the data
    loaded_embs, loaded_qids = load_embs_and_qids_with_normalization(tmp_path)

    assert loaded_embs.shape == test_embs.shape
    assert loaded_qids.shape == test_qids.shape
    assert np.allclose(np.linalg.norm(loaded_embs, axis=1), 1.0)


def test_get_scann_searcher(dummy_data):
    embs, qids = dummy_data
    searcher = get_scann_searcher(embs, qids)
    assert isinstance(searcher, ScaNNSearcher)


def test_get_brute_force_searcher(dummy_data):
    embs, qids = dummy_data
    searcher = get_brute_force_searcher(embs, qids)
    assert isinstance(searcher, BruteForceSearcher)


def test_get_faiss_searcher(dummy_data):
    embs, qids = dummy_data
    searcher = get_faiss_searcher(embs, qids)
    assert isinstance(searcher, FaissSearcher)


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
    with patch(
        "finetunings.evaluation.find_recall.load_qids_remap", return_value=mock_qid_map
    ):
        # Test qids_remap function
        input_qids = np.array([1, 2, 3, 4, 5])
        expected_output = np.array([10, 20, 30, 4, 5])

        remapped_qids = qids_remap(input_qids, "dummy_path")

        assert np.array_equal(remapped_qids, expected_output)
