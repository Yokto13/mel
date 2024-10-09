from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import gin
from multilingual_dataset.mixer import Mixer
from utils.qids_remap import remap_qids_decorator, qids_remap

gin.add_config_file_search_path("configs/general.gin")


def mock_remap_qids(qids, _):
    return qids


@pytest.fixture
def create_dummy_npz_files(tmpdir):
    file_paths = []
    for i in range(10):
        file_path = Path(tmpdir) / f"mentions_{i}.npz"
        tokens = np.random.randint(1, 1000, size=(100, 10))  # 100 rows, 10 columns
        qids = np.random.randint(1, 1000, size=(100,))
        np.savez(file_path, tokens=tokens, qids=qids)
        file_paths.append(file_path)
    return file_paths


def load_npz_content(file_path):
    with np.load(file_path) as data:
        return data["tokens"], data["qids"]


@patch("utils.qids_remap.qids_remap", side_effect=mock_remap_qids)
def test_mix_changes_file_contents(mock_qids_remap, create_dummy_npz_files):
    file_paths = create_dummy_npz_files
    original_contents = [load_npz_content(path) for path in file_paths]

    mixer = Mixer(buffer_size=10)
    mixer.mix(file_paths, n_of_mixings=1, compress_output=False)

    new_contents = [load_npz_content(path) for path in file_paths]

    assert not all(
        np.array_equal(orig[0], new[0]) and np.array_equal(orig[1], new[1])
        for orig, new in zip(original_contents, new_contents)
    )


@patch("utils.qids_remap.qids_remap", side_effect=mock_remap_qids)
def test_mix_preserves_total_content(mock_qids_remap, create_dummy_npz_files):
    file_paths = create_dummy_npz_files
    original_tokens = np.concatenate([load_npz_content(path)[0] for path in file_paths])
    original_qids = np.concatenate([load_npz_content(path)[1] for path in file_paths])

    mixer = Mixer(buffer_size=10)
    mixer.mix(file_paths, n_of_mixings=1, compress_output=False)

    new_tokens = np.concatenate([load_npz_content(path)[0] for path in file_paths])
    new_qids = np.concatenate([load_npz_content(path)[1] for path in file_paths])

    assert np.array_equal(
        np.sort(original_tokens.flatten()), np.sort(new_tokens.flatten())
    )
    assert np.array_equal(np.sort(original_qids), np.sort(new_qids))


@patch("utils.qids_remap.qids_remap", side_effect=mock_remap_qids)
def test_mix_multiple_times(mock_qids_remap, create_dummy_npz_files):
    file_paths = create_dummy_npz_files
    original_contents = [load_npz_content(path) for path in file_paths]

    mixer = Mixer(buffer_size=10)
    mixer.mix(file_paths, n_of_mixings=3, compress_output=False)

    new_contents = [load_npz_content(path) for path in file_paths]

    assert not all(
        np.array_equal(orig[0], new[0]) and np.array_equal(orig[1], new[1])
        for orig, new in zip(original_contents, new_contents)
    )


@patch("utils.qids_remap.qids_remap", side_effect=mock_remap_qids)
def test_mix_with_small_buffer(mock_qids_remap, create_dummy_npz_files):
    file_paths = create_dummy_npz_files
    original_contents = [load_npz_content(path) for path in file_paths]

    mixer = Mixer(buffer_size=2)
    mixer.mix(file_paths, n_of_mixings=1, compress_output=False)

    new_contents = [load_npz_content(path) for path in file_paths]

    assert not all(
        np.array_equal(orig[0], new[0]) and np.array_equal(orig[1], new[1])
        for orig, new in zip(original_contents, new_contents)
    )


@patch("utils.qids_remap.qids_remap", side_effect=mock_remap_qids)
def test_mix_empty_file_list(mock_qids_remap):
    mixer = Mixer(buffer_size=1000)
    mixer.mix([], n_of_mixings=1, compress_output=False)
    # This test passes if no exception is raised


@patch("utils.qids_remap.qids_remap", side_effect=mock_remap_qids)
def test_mix_single_file(mock_qids_remap, tmp_path):
    file_path = tmp_path / "mentions_0.npz"
    tokens = np.random.randint(1, 1000, size=(100, 10))
    qids = np.random.randint(1, 1000, size=(100,))
    qids = qids_remap(
        qids,
    )
    np.savez_compressed(file_path, tokens=tokens, qids=qids)

    mixer = Mixer(buffer_size=1000)
    mixer.mix([file_path], n_of_mixings=1, compress_output=False)

    new_tokens, new_qids = load_npz_content(file_path)
    assert not np.array_equal(tokens, new_tokens)
    assert not np.array_equal(qids, new_qids)
    assert set(qids) == set(new_qids)


@patch("utils.qids_remap.qids_remap", side_effect=mock_remap_qids)
def test_mix_preserves_consistency(mock_qids_remap, create_dummy_npz_files):
    file_paths = create_dummy_npz_files
    original_shapes = [load_npz_content(path)[0].shape for path in file_paths]

    mixer = Mixer(buffer_size=1000)
    mixer.mix(file_paths, n_of_mixings=1, compress_output=False)

    for path, original_shape in zip(file_paths, original_shapes):
        tokens, qids = load_npz_content(path)
        assert len(qids) == tokens.shape[0]


@patch("utils.qids_remap.qids_remap", side_effect=mock_remap_qids)
def test_mix_compress(mock_qids_remap, tmp_path):
    file_path_1 = tmp_path / "mentions_0.npz"
    file_path_2 = tmp_path / "mentions_1.npz"
    tokens = np.random.randint(1, 1000, size=(100, 10))
    qids = np.random.randint(1, 1000, size=(100,))
    np.savez(file_path_1, tokens=tokens, qids=qids)
    np.savez(file_path_2, tokens=tokens, qids=qids)

    mixer = Mixer(buffer_size=1000)
    mixer.mix([file_path_1], n_of_mixings=1, compress_output=True)

    mixer.mix([file_path_2], n_of_mixings=1, compress_output=False)

    assert file_path_1.exists()
    assert file_path_2.exists()

    assert file_path_1.stat().st_size < file_path_2.stat().st_size
