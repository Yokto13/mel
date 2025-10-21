import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix

from utils.loaders import (
    AliasTableLoader,
    load_embs_and_qids,
    load_embs_qids_tokens,
    load_qids,
    load_qids_npy,
    load_tokens_qids,
    load_tokens_qids_from_dir,
    map_qids_to_token_matrix,
)


def mock_remap_qids(qids, _):
    return qids


def _create_tokens_qids_npz(dir_path: Path, file_name: str, tokens: np.ndarray, qids: np.ndarray):
    file_path = Path(dir_path) / file_name
    np.savez_compressed(file_path, tokens=np.array(tokens), qids=np.array(qids))
    return file_path


@patch("utils.qids_remap.qids_remap", side_effect=mock_remap_qids)
def test_load_tokens_qids_with_path_object(mock_qids_remap):
    with tempfile.TemporaryDirectory() as temp_dir:
        file_path = Path(temp_dir) / "mentions_2.npz"

        test_tokens = np.array([[1, 2, 3], [4, 5, 6]])
        test_qids = np.array([200, 100])

        np.savez_compressed(file_path, tokens=test_tokens, qids=test_qids)

        loaded_tokens, loaded_qids = load_tokens_qids(file_path)

        assert np.array_equal(loaded_tokens, test_tokens)
        assert np.array_equal(loaded_qids, test_qids)


@patch("utils.qids_remap.qids_remap", side_effect=mock_remap_qids)
def test_load_tokens_qids_with_string_path(mock_qids_remap):
    with tempfile.TemporaryDirectory() as temp_dir:
        file_path = str(Path(temp_dir) / "mentions_1.npz")

        test_tokens = np.array([[10, 20], [30, 40], [50, 60]])
        test_qids = np.array([1000, 3000, 2000])

        np.savez_compressed(file_path, tokens=test_tokens, qids=test_qids)

        loaded_tokens, loaded_qids = load_tokens_qids(file_path)

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
def test_embs_qids_loaders(mock_qids_remap, loader_func, file_name, test_data, use_string_path):
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
def test_loaders_sort(mock_qids_remap, loader_func, file_name, test_data, use_string_path):
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

        qid_emb_test_data = {qid: emb for qid, emb in zip(test_data["qids"], test_data["embs"])}

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
def test_loaders_sort_stable(mock_qids_remap, loader_func, file_name, test_data, use_string_path):
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


@patch("utils.qids_remap.qids_remap", side_effect=mock_remap_qids)
def test_load_tokens_qids_from_dir_single_file(mock_qids_remap):
    with tempfile.TemporaryDirectory() as temp_dir:
        dir_path = Path(temp_dir)

        tokens = np.array([[1, 2, 3], [4, 5, 6]])
        qids = np.array([10, 20])
        _create_tokens_qids_npz(dir_path, "data_0.npz", tokens, qids)

        loaded_tokens, loaded_qids = load_tokens_qids_from_dir(dir_path)

        sort_indices = np.argsort(qids, kind="stable")
        assert np.array_equal(loaded_qids, qids[sort_indices])
        assert np.array_equal(loaded_tokens, tokens[sort_indices])


@patch("utils.qids_remap.qids_remap", side_effect=mock_remap_qids)
def test_load_tokens_qids_from_dir_multiple_files(mock_qids_remap):
    with tempfile.TemporaryDirectory() as temp_dir:
        dir_path = Path(temp_dir)

        tokens_a = np.array([[1, 1, 1], [2, 2, 2]])
        qids_a = np.array([20, 10])
        tokens_b = np.array([[3, 3, 3]])
        qids_b = np.array([30])
        _create_tokens_qids_npz(dir_path, "data_a.npz", tokens_a, qids_a)
        _create_tokens_qids_npz(dir_path, "data_b.npz", tokens_b, qids_b)

        loaded_tokens, loaded_qids = load_tokens_qids_from_dir(dir_path)

        expected_tokens = np.vstack([tokens_a, tokens_b])
        expected_qids = np.concatenate([qids_a, qids_b])
        sort_indices = np.argsort(expected_qids, kind="stable")

        assert np.array_equal(loaded_qids, expected_qids[sort_indices])
        assert np.array_equal(loaded_tokens, expected_tokens[sort_indices])


@patch("utils.qids_remap.qids_remap", side_effect=mock_remap_qids)
def test_load_tokens_qids_from_dir_max_limit_overflow(mock_qids_remap):
    with tempfile.TemporaryDirectory() as temp_dir:
        dir_path = Path(temp_dir)

        tokens_a = np.array([[1, 0], [2, 0], [3, 0]])
        qids_a = np.array([300, 100, 200])
        tokens_b = np.array([[4, 0], [5, 0], [6, 0]])
        qids_b = np.array([600, 500, 400])
        _create_tokens_qids_npz(dir_path, "data_a.npz", tokens_a, qids_a)
        _create_tokens_qids_npz(dir_path, "data_b.npz", tokens_b, qids_b)

        max_items = 4
        loaded_tokens, loaded_qids = load_tokens_qids_from_dir(
            dir_path, max_items_to_load=max_items
        )

        expected_tokens = np.vstack([tokens_a, tokens_b])
        expected_qids = np.concatenate([qids_a, qids_b])
        sort_indices = np.argsort(expected_qids, kind="stable")

        assert len(loaded_tokens) == len(expected_tokens)
        assert len(loaded_tokens) > max_items
        assert np.array_equal(loaded_qids, expected_qids[sort_indices])
        assert np.array_equal(loaded_tokens, expected_tokens[sort_indices])


@patch("utils.qids_remap.qids_remap", side_effect=mock_remap_qids)
def test_load_tokens_qids_from_dir_large_max_limit_loads_all(mock_qids_remap):
    with tempfile.TemporaryDirectory() as temp_dir:
        dir_path = Path(temp_dir)

        tokens_a = np.array([[10, 10], [20, 20]])
        qids_a = np.array([200, 100])
        tokens_b = np.array([[30, 30]])
        qids_b = np.array([300])
        tokens_c = np.array([[40, 40]])
        qids_c = np.array([400])
        _create_tokens_qids_npz(dir_path, "data_a.npz", tokens_a, qids_a)
        _create_tokens_qids_npz(dir_path, "data_b.npz", tokens_b, qids_b)
        _create_tokens_qids_npz(dir_path, "data_c.npz", tokens_c, qids_c)

        max_items = 100
        loaded_tokens, loaded_qids = load_tokens_qids_from_dir(
            dir_path, max_items_to_load=max_items
        )

        expected_tokens = np.vstack([tokens_a, tokens_b, tokens_c])
        expected_qids = np.concatenate([qids_a, qids_b, qids_c])
        sort_indices = np.argsort(expected_qids, kind="stable")

        assert len(loaded_tokens) == len(expected_tokens)
        assert len(loaded_tokens) < max_items
        assert np.array_equal(loaded_qids, expected_qids[sort_indices])
        assert np.array_equal(loaded_tokens, expected_tokens[sort_indices])


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


@pytest.mark.parametrize("use_string_path", [True, False])
@patch("utils.qids_remap.qids_remap", side_effect=mock_remap_qids)
def test_load_qids_npy(mock_qids_remap, use_string_path: bool) -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        dir_path = Path(temp_dir)
        if use_string_path:
            dir_path = str(dir_path)
        file_path = Path(dir_path) / "qids.npy"

        test_qids = np.array([100, 200, 300])

        np.save(file_path, test_qids)

        loaded_qids = load_qids_npy(file_path)

        assert np.array_equal(loaded_qids, test_qids)
        assert isinstance(loaded_qids, np.ndarray)


def test_map_qids_to_token_matrix() -> None:
    """
    Tests that qids are correctly mapped to their token vectors in a sparse matrix.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        dir_path = Path(temp_dir)

        tokens = np.array([[1.0, 1.5], [2.0, 2.5], [3.0, 3.5]])
        qids = np.array([300, 100, 200])
        _create_tokens_qids_npz(dir_path, "tokens_qids.npz", tokens, qids)

        token_matrix = map_qids_to_token_matrix(dir_path)

        assert isinstance(token_matrix, csr_matrix)
        assert token_matrix.shape == (301, 2)

        for i, qid in enumerate(qids):
            original_vector = tokens[i]
            retrieved_vector = token_matrix[qid].toarray()[0]

            assert np.array_equal(original_vector, retrieved_vector)

        non_existent_qid = 150
        zero_vector = token_matrix[non_existent_qid].toarray()[0]
        assert np.array_equal(zero_vector, np.zeros(tokens.shape[1]))


@pytest.mark.parametrize("lowercase", [True, False])
class TestAliasTableLoader:
    def setup_method(self, lowercase):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.mewsli_root_path = Path(self.temp_dir.name) / "mewsli"
        self.damuel_root_path = Path(self.temp_dir.name) / "damuel"
        self.loader = AliasTableLoader(
            mewsli_root_path=self.mewsli_root_path,
            damuel_root_path=self.damuel_root_path,
            lowercase=lowercase,
        )
        self.lowercase = lowercase

    def teardown_method(self):
        self.temp_dir.cleanup()

    def create_dummy_mentions_tsv(self, file_path: Path):
        data = {
            "mention": ["mention1", "Mention2", "mention3"],
            "qid": ["Q123", "Q456", "Q789"],
        }
        df = pd.DataFrame(data)
        df.to_csv(file_path, sep="\t", index=False)

    @patch("utils.qids_remap.qids_remap", side_effect=mock_remap_qids)
    def test_load_mentions_with_path_object(self, mock_qids_remap, lowercase):
        file_path = self.mewsli_root_path / "ar" / "mentions.tsv"
        file_path.parent.mkdir(parents=True)

        self.create_dummy_mentions_tsv(file_path)

        mentions, qids = self.loader.load_mewsli("ar")
        expected_mentions = (
            ["mention1", "mention2", "mention3"]
            if self.lowercase
            else ["mention1", "Mention2", "mention3"]
        )
        assert mentions == expected_mentions
        assert list(qids) == [123, 456, 789]

    def create_dummy_damuel_dir(self, dir_path: Path):
        dir_path.mkdir(parents=True)
        data = [
            ("mention1", 123),
            ("mention2", 456),
            ("Mention3", 789),
        ]
        for i, (mention, qid) in enumerate(data):
            file_path = dir_path / f"alias_{i}.txt"
            with file_path.open("w") as f:
                f.write(f"{mention}\t{qid}\n")

    @patch("utils.qids_remap.qids_remap", side_effect=mock_remap_qids)
    @patch("utils.loaders.run_alias_table_damuel")
    def test_load_damuel(self, mock_pipeline, mock_qids_remap, lowercase):
        lang = "en"
        damuel_dir = self.damuel_root_path / f"dataset_{lang}"
        self.create_dummy_damuel_dir(damuel_dir)

        mock_pipeline.return_value = [
            [
                ("mention1", 123),
                ("mention2", 456),
                ("Mention3", 789),
            ]
        ]

        mentions, qids = self.loader.load_damuel(lang)

        expected_mentions = (
            ["mention1", "mention2", "mention3"]
            if self.lowercase
            else ["mention1", "mention2", "Mention3"]
        )
        assert mentions == expected_mentions
        assert list(qids) == [123, 456, 789]

    def create_dummy_damuel_subdirs(self):
        (self.damuel_root_path / "dataset_en").mkdir(parents=True)
        (self.damuel_root_path / "dataset_fr").mkdir(parents=True)

    def test_construct_damuel_path(self, lowercase):
        self.create_dummy_damuel_subdirs()

        lang = "en"
        expected_path = (self.damuel_root_path / f"dataset_{lang}").as_posix()
        constructed_path = self.loader._construct_damuel_path(lang)
        assert constructed_path == expected_path

        with pytest.raises(
            FileNotFoundError,
            match=f"No directory ending with 'de' found in {self.damuel_root_path}",
        ):
            self.loader._construct_damuel_path("de")
