import numpy as np
import pytest
from finetunings.generate_epochs.generate import reorder_data_to_match_qids


def test_reorder_data_to_match_qids_basic_reordering():
    tokens = np.array(["a", "b", "c", "d"])
    wrong_qids = np.array([1, 2, 3, 4])
    correct_qids = np.array([2, 4, 1, 3])
    expected_result = np.array(["b", "d", "a", "c"])

    result = reorder_data_to_match_qids(tokens, wrong_qids, correct_qids)
    np.testing.assert_array_equal(result, expected_result)


def test_reorder_data_to_match_qids_raises_exception():
    tokens = np.array(["a", "b", "c", "d"])
    wrong_qids = np.array([1, 2, 3, 4])
    correct_qids = np.array([1, 2, 3, 5])  # 5 is different from 4

    with pytest.raises(ValueError, match="Qids contain different elements"):
        reorder_data_to_match_qids(tokens, wrong_qids, correct_qids)


def test_reorder_data_to_match_qids_identical_qids():
    tokens = np.array(["a", "b", "c", "d"])
    qids = np.array([1, 2, 3, 4])

    result = reorder_data_to_match_qids(tokens, qids, qids)
    np.testing.assert_array_equal(result, tokens)
