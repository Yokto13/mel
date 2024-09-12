import pytest
import numpy as np
from pathlib import Path
import tempfile
import os

from multilingual_dataset.mixer import (
    Mixer,
)


@pytest.fixture
def create_dummy_npz_files():
    with tempfile.TemporaryDirectory() as temp_dir:
        file_paths = []
        for i in range(50):
            file_path = Path(temp_dir) / f"mentions_{i}.npz"
            tokens = np.random.randint(
                1, 1000, size=(10000, 10)
            )  # 10000 rows, 10 columns
            qids = np.random.randint(1, 1000, size=(10000,))
            np.savez_compressed(file_path, tokens=tokens, qids=qids)
            file_paths.append(file_path)
        yield file_paths


def load_npz_content(file_path):
    with np.load(file_path) as data:
        return data["tokens"], data["qids"]


def test_mix_changes_file_contents(create_dummy_npz_files):
    file_paths = create_dummy_npz_files
    original_contents = [load_npz_content(path) for path in file_paths]

    mixer = Mixer(buffer_size=10)
    mixer.mix(file_paths, n_of_mixings=1)

    new_contents = [load_npz_content(path) for path in file_paths]

    assert not all(
        np.array_equal(orig[0], new[0]) and np.array_equal(orig[1], new[1])
        for orig, new in zip(original_contents, new_contents)
    )


def test_mix_preserves_total_content(create_dummy_npz_files):
    file_paths = create_dummy_npz_files
    original_tokens = np.concatenate([load_npz_content(path)[0] for path in file_paths])
    original_qids = np.concatenate([load_npz_content(path)[1] for path in file_paths])

    mixer = Mixer(buffer_size=10)
    mixer.mix(file_paths, n_of_mixings=1)

    new_tokens = np.concatenate([load_npz_content(path)[0] for path in file_paths])
    new_qids = np.concatenate([load_npz_content(path)[1] for path in file_paths])

    assert np.array_equal(
        np.sort(original_tokens.flatten()), np.sort(new_tokens.flatten())
    )
    assert np.array_equal(np.sort(original_qids), np.sort(new_qids))


def test_mix_multiple_times(create_dummy_npz_files):
    file_paths = create_dummy_npz_files
    original_contents = [load_npz_content(path) for path in file_paths]

    mixer = Mixer(buffer_size=10)
    mixer.mix(file_paths, n_of_mixings=3)

    new_contents = [load_npz_content(path) for path in file_paths]

    assert not all(
        np.array_equal(orig[0], new[0]) and np.array_equal(orig[1], new[1])
        for orig, new in zip(original_contents, new_contents)
    )


def test_mix_with_small_buffer(create_dummy_npz_files):
    file_paths = create_dummy_npz_files
    original_contents = [load_npz_content(path) for path in file_paths]

    mixer = Mixer(buffer_size=2)
    mixer.mix(file_paths, n_of_mixings=1)

    new_contents = [load_npz_content(path) for path in file_paths]

    assert not all(
        np.array_equal(orig[0], new[0]) and np.array_equal(orig[1], new[1])
        for orig, new in zip(original_contents, new_contents)
    )


def test_mix_empty_file_list():
    mixer = Mixer(buffer_size=1000)
    mixer.mix([], n_of_mixings=1)
    # This test passes if no exception is raised


def test_mix_single_file(tmp_path):
    file_path = tmp_path / "mentions_0.npz"
    tokens = np.random.randint(1, 1000, size=(100, 10))
    qids = np.random.randint(1, 1000, size=(100,))
    np.savez_compressed(file_path, tokens=tokens, qids=qids)

    mixer = Mixer(buffer_size=1000)
    mixer.mix([file_path], n_of_mixings=1)

    new_tokens, new_qids = load_npz_content(file_path)
    assert not np.array_equal(tokens, new_tokens)
    assert not np.array_equal(qids, new_qids)
    assert set(qids) == set(new_qids)


def test_mix_preserves_consistency(create_dummy_npz_files):
    file_paths = create_dummy_npz_files
    original_shapes = [load_npz_content(path)[0].shape for path in file_paths]

    mixer = Mixer(buffer_size=1000)
    mixer.mix(file_paths, n_of_mixings=1)

    for path, original_shape in zip(file_paths, original_shapes):
        tokens, qids = load_npz_content(path)
        assert len(qids) == tokens.shape[0]


def test_mix_compress(tmp_path):
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
