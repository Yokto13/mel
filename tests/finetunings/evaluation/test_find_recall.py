import json
from unittest.mock import patch

import numpy as np
import pytest
from pathlib import Path
from unittest.mock import Mock

from finetunings.evaluation.find_recall import (
    get_brute_force_searcher,
    get_faiss_searcher,
    get_scann_searcher,
    load_embs_and_qids_with_normalization,
    find_candidates,
)
from models.searchers.brute_force_searcher import BruteForceSearcher
from models.searchers.faiss_searcher import FaissSearcher
from models.searchers.scann_searcher import ScaNNSearcher


def mock_remap_qids(qids, _):
    return qids


@pytest.fixture(scope="module")
def dummy_data():
    embs = np.random.rand(100000, 100).astype(np.float32)
    qids = np.arange(100000, dtype=np.int64)
    return embs, qids


@patch("utils.qids_remap.qids_remap", side_effect=mock_remap_qids)
def test_load_embs_and_qids_with_normalization(mock_qids_remap, tmp_path):
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


@pytest.mark.slow
def test_get_scann_searcher(dummy_data):
    embs, qids = dummy_data
    searcher = get_scann_searcher(embs, qids)
    assert isinstance(searcher, ScaNNSearcher)


@pytest.mark.slow
def test_get_brute_force_searcher(dummy_data):
    embs, qids = dummy_data
    searcher = get_brute_force_searcher(embs, qids)
    assert isinstance(searcher, BruteForceSearcher)


@pytest.mark.slow
def test_get_faiss_searcher(dummy_data):
    embs, qids = dummy_data
    searcher = get_faiss_searcher(embs, qids)
    assert isinstance(searcher, FaissSearcher)


def test_find_candidates(tmp_path: Path) -> None:
    mock_embs = np.random.rand(10, 128)
    mock_qids = np.arange(10)
    mock_candidate_qids = np.array([np.arange(5) for _ in range(10)])
    mock_recall_value = 0.75

    candidates_path = str(tmp_path / "candidates.npz")

    # Mock the dependencies
    with patch(
        "finetunings.evaluation.find_recall.load_embs_and_qids_with_normalization"
    ) as mock_load, patch(
        "finetunings.evaluation.find_recall.BruteForceSearcher"
    ) as mock_searcher_cls, patch(
        "finetunings.evaluation.find_recall.RecallCalculator"
    ) as mock_rc_cls:

        # Setup the mocks
        mock_load.side_effect = [(mock_embs, mock_qids), (mock_embs, mock_qids)]
        mock_searcher = Mock()
        mock_searcher_cls.return_value = mock_searcher
        mock_rc = Mock()
        mock_rc.recall.return_value = (mock_recall_value, mock_candidate_qids)
        mock_rc_cls.return_value = mock_rc

        # Call the function
        find_candidates(
            damuel_entities="dummy_damuel.npz",
            candidates_path=candidates_path,
            mewsli="dummy_mewsli.npz",
            recall=5,
        )

        # Verify the file was saved
        assert Path(candidates_path).exists()

        # Load and verify the saved data
        saved_data = np.load(candidates_path)
        np.testing.assert_array_equal(saved_data["candidate_qids"], mock_candidate_qids)
