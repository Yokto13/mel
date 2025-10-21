from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix

from scripts.reranking.change_dataset_tokens import process_directory, update_tokens_in_file


def _create_npz_file(path: Path, filename: str, **kwargs):
    """Helper to create an .npz file for testing."""
    np.savez(path / filename, **kwargs)


def _create_sparse_matrix(data_dict: dict, shape: tuple[int, int], dtype=np.float32) -> csr_matrix:
    """Helper to build a CSR matrix from a dictionary of {qid: vector}."""
    rows, cols, data = [], [], []
    vector_len = shape[1]
    for qid, vector in data_dict.items():
        rows.extend([qid] * vector_len)
        cols.extend(range(vector_len))
        data.extend(vector)
    return coo_matrix((data, (rows, cols)), shape=shape, dtype=dtype).tocsr()


def test_update_tokens_when_match_exists(tmp_path: Path):
    """
    Tests that the file is modified.
    """
    test_file = tmp_path / "data.npz"
    original_tokens = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
    original_qids = np.array([100, 200, 300])
    _create_npz_file(
        tmp_path, "data.npz", description_tokens=original_tokens.copy(), qids=original_qids
    )

    new_token_vector = np.array([99.0, 99.0])
    update_matrix = _create_sparse_matrix(
        {100: original_tokens[0], 200: new_token_vector, 300: original_tokens[2]}, shape=(400, 2)
    )
    update_tokens_in_file(test_file, update_matrix)

    loaded_data = np.load(test_file)
    updated_tokens = loaded_data["description_tokens"]

    assert np.array_equal(updated_tokens[1], new_token_vector)
    assert np.array_equal(updated_tokens[0], original_tokens[0])
    assert np.array_equal(updated_tokens[2], original_tokens[2])


@patch("scripts.reranking.change_dataset_tokens.update_tokens_in_file")
@patch("scripts.reranking.change_dataset_tokens.map_qids_to_token_matrix")
def test_process_directory_orchestration(
    mock_map_qids: MagicMock, mock_update_file: MagicMock, tmp_path: Path
):
    dataset_dir = tmp_path / "dataset"
    tokens_dir = tmp_path / "tokens"
    dataset_dir.mkdir()
    tokens_dir.mkdir()

    file_paths = [dataset_dir / "a.npz", dataset_dir / "b.npz", dataset_dir / "c.npz"]

    mock_map_qids.return_value = csr_matrix((3, 2))
    mock_update_file.side_effect = [True, False, True]

    with patch.object(Path, "glob", return_value=file_paths):
        process_directory(dataset_dir, tokens_dir)

    mock_map_qids.assert_called_once_with(tokens_dir, verbose=True)
    assert mock_update_file.call_count == len(file_paths)
