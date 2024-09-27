import json
from unittest.mock import patch

import numpy as np
import pytest

from finetunings.evaluation.find_recall import (
    get_brute_force_searcher,
    get_faiss_searcher,
    get_scann_searcher,
    load_embs_and_qids_with_normalization,
)
from models.searchers.brute_force_searcher import BruteForceSearcher
from models.searchers.faiss_searcher import FaissSearcher
from models.searchers.scann_searcher import ScaNNSearcher


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
