import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from utils.loaders import (
    load_embs_and_qids,
    load_embs_qids_tokens,
    load_mentions,
    load_qids,
)


def mock_remap_qids(qids, _):
    return qids


@patch("utils.qids_remap.qids_remap", side_effect=mock_remap_qids)
def test_load_mentions_with_path_object(mock_qids_remap):
    with tempfile.TemporaryDirectory() as temp_dir:
        file_path = Path(temp_dir) / "mentions_2.npz"

        test_tokens = np.array([[1, 2, 3], [4, 5, 6]])
        test_qids = np.array([200, 100])

        np.savez_compressed(file_path, tokens=test_tokens, qids=test_qids)

        loaded_tokens, loaded_qids = load_mentions(file_path)

        assert np.array_equal(loaded_tokens, test_tokens)
        assert np.array_equal(loaded_qids, test_qids)


@patch("utils.qids_remap.qids_remap", side_effect=mock_remap_qids)
def test_load_mentions_with_string_path(mock_qids_remap):
    with tempfile.TemporaryDirectory() as temp_dir:
        file_path = str(Path(temp_dir) / "mentions_1.npz")

        test_tokens = np.array([[10, 20], [30, 40], [50, 60]])
        test_qids = np.array([1000, 3000, 2000])

        np.savez_compressed(file_path, tokens=test_tokens, qids=test_qids)

        loaded_tokens, loaded_qids = load_mentions(file_path)

        assert np.array_equal(loaded_tokens, test_tokens)
        assert np.array_equal(loaded_qids, test_qids)


@pytest.mark.parametrize(
    "loader_func, file_name, test_data",
    [
        (
            load_embs_and_qids,
            "embs_qids.npz",
            {
                "embs": np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
                "qids": np.array([300, 100, 200]),
            },
        ),
        (
            load_embs_qids_tokens,
            "embs_qids_tokens.npz",
            {
                "embs": np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
                "qids": np.array([300, 100, 200]),
                "tokens": np.array([[1, 2], [3, 4], [5, 6]]),
            },
        ),
    ],
)
@pytest.mark.parametrize("use_string_path", [True, False])
@patch("utils.qids_remap.qids_remap", side_effect=mock_remap_qids)
def test_embs_qids_loaders(
    mock_qids_remap, loader_func, file_name, test_data, use_string_path
):
    with tempfile.TemporaryDirectory() as temp_dir:
        dir_path = Path(temp_dir)
        if use_string_path:
            dir_path = str(dir_path)
        file_path = Path(dir_path) / file_name

        np.savez_compressed(file_path, **test_data)

        loaded_data = loader_func(dir_path)

        for i, (loaded, original) in enumerate(zip(loaded_data, test_data.values())):
            assert np.array_equal(loaded, original)
            assert isinstance(loaded, np.ndarray)

        assert len(loaded_data) == len(test_data)


@pytest.mark.parametrize(
    "loader_func, file_name, test_data",
    [
        (
            load_embs_and_qids,
            "embs_qids.npz",
            {
                "embs": np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
                "qids": np.array([300, 100, 200]),
            },
        ),
        (
            load_embs_qids_tokens,
            "embs_qids_tokens.npz",
            {
                "embs": np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
                "qids": np.array([300, 100, 200]),
                "tokens": np.array([[1, 2], [3, 4], [5, 6]]),
            },
        ),
    ],
)
@pytest.mark.parametrize("use_string_path", [True, False])
@pytest.mark.skip(
    reason="Sorting is currently disabled because it interferes with MultifileDataset"
)
@patch("utils.qids_remap.qids_remap", side_effect=mock_remap_qids)
def test_loaders_sort(
    mock_qids_remap, loader_func, file_name, test_data, use_string_path
):
    with tempfile.TemporaryDirectory() as temp_dir:
        dir_path = Path(temp_dir)
        if use_string_path:
            dir_path = str(dir_path)
        file_path = Path(dir_path) / file_name

        np.savez_compressed(file_path, **test_data)

        loaded_data = loader_func(dir_path)

        sort_indices = np.argsort(test_data["qids"])

        for i, (loaded, original) in enumerate(zip(loaded_data, test_data.values())):
            assert np.array_equal(loaded, original[sort_indices])
            assert isinstance(loaded, np.ndarray)

        assert len(loaded_data) == len(test_data)


@pytest.mark.parametrize(
    "loader_func, file_name, test_data",
    [
        (
            load_embs_and_qids,
            "embs_qids.npz",
            {
                "embs": np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
                "qids": np.array([300, 100, 200]),
            },
        ),
        (
            load_embs_qids_tokens,
            "embs_qids_tokens.npz",
            {
                "embs": np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
                "qids": np.array([300, 100, 200]),
                "tokens": np.array([[1, 2], [3, 4], [5, 6]]),
            },
        ),
    ],
)
@pytest.mark.parametrize("use_string_path", [True, False])
@pytest.mark.skip(
    reason="Sorting is currently disabled because it interferes with MultifileDataset"
)
@patch("utils.qids_remap.qids_remap", side_effect=mock_remap_qids)
def test_loaders_sort_corresponding(
    mock_qids_remap, loader_func, file_name, test_data, use_string_path
):
    with tempfile.TemporaryDirectory() as temp_dir:
        dir_path = Path(temp_dir)
        if use_string_path:
            dir_path = str(dir_path)
        file_path = Path(dir_path) / file_name

        np.savez_compressed(file_path, **test_data)

        loaded_data = loader_func(dir_path)

        qid_emb_test_data = {
            qid: emb for qid, emb in zip(test_data["qids"], test_data["embs"])
        }

        for emb, qid in zip(loaded_data[0], loaded_data[1]):
            assert np.array_equal(emb, qid_emb_test_data[qid])


@pytest.mark.parametrize(
    "loader_func, file_name, test_data",
    [
        (
            load_embs_and_qids,
            "embs_qids.npz",
            {
                "embs": np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
                "qids": np.array([300, 100, 200]),
            },
        ),
        (
            load_embs_qids_tokens,
            "embs_qids_tokens.npz",
            {
                "embs": np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
                "qids": np.array([300, 100, 200]),
                "tokens": np.array([[1, 2], [3, 4], [5, 6]]),
            },
        ),
    ],
)
@pytest.mark.parametrize("use_string_path", [True, False])
@pytest.mark.skip(
    reason="Sorting is currently disabled because it interferes with MultifileDataset"
)
@patch("utils.qids_remap.qids_remap", side_effect=mock_remap_qids)
def test_loaders_sort_stable(
    mock_qids_remap, loader_func, file_name, test_data, use_string_path
):
    with tempfile.TemporaryDirectory() as temp_dir:
        dir_path = Path(temp_dir)
        if use_string_path:
            dir_path = str(dir_path)
        file_path = Path(dir_path) / file_name

        np.savez_compressed(file_path, **test_data)

        loaded_data = loader_func(dir_path)
        loaded_data2 = loader_func(dir_path)
        loaded_data3 = loader_func(dir_path)

        print(loaded_data[0])
        print(loaded_data[1])
        print(loaded_data2[0])
        print(loaded_data2[1])
        print(loaded_data3[0])
        print(loaded_data3[1])
        for i in range(len(loaded_data)):
            assert np.array_equal(loaded_data[i], loaded_data2[i])
            assert np.array_equal(loaded_data[i], loaded_data3[i])


@pytest.mark.parametrize("use_string_path", [True, False])
@pytest.mark.parametrize(
    "file_name",
    [
        "embs_qids_tokens.npz",
        "embs_qids_tokens_10.npz",
        "embs_qids_tokens_0.npz",
        "blabla.npz",
    ],
)
@patch("utils.qids_remap.qids_remap", side_effect=mock_remap_qids)
def test_embs_qids_tokens_from_file(mock_qids_remap, use_string_path, file_name):
    with tempfile.TemporaryDirectory() as temp_dir:
        dir_path = Path(temp_dir)
        if use_string_path:
            dir_path = str(dir_path)
        file_path = Path(dir_path) / file_name

        test_data = {
            "embs": np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
            "qids": np.array([300, 100, 200]),
            "tokens": np.array([[1, 2], [3, 4], [5, 6]]),
        }

        np.savez_compressed(file_path, **test_data)

        loaded_data = load_embs_qids_tokens(file_path)

        for i, (loaded, original) in enumerate(zip(loaded_data, test_data.values())):
            assert np.array_equal(loaded, original)
            assert isinstance(loaded, np.ndarray)

        assert len(loaded_data) == len(test_data)


@pytest.mark.parametrize("use_string_path", [True, False])
@patch("utils.qids_remap.qids_remap", side_effect=mock_remap_qids)
def test_load_qids(mock_qids_remap, use_string_path: bool) -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        dir_path = Path(temp_dir)
        if use_string_path:
            dir_path = str(dir_path)
        file_path = Path(dir_path) / "qids.npz"

        test_qids = np.array([100, 200, 300])

        np.savez(file_path, qids=test_qids)

        loaded_qids = load_qids(file_path)

        assert np.array_equal(loaded_qids, test_qids)
        assert isinstance(loaded_qids, np.ndarray)
