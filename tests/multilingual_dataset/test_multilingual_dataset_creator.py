from collections import Counter, defaultdict
import numpy as np
import pytest
from pathlib import Path
from unittest.mock import Mock, patch

from multilingual_dataset.creator import (
    _LinksCreator,
    _KBCreator,
    MultilingualDatasetCreator,
    DamuelPaths,
)


@pytest.fixture
def sample_langs():
    return ["en", "fr", "de"]


@pytest.fixture
def setup_teardown(tmpdir, sample_langs):
    for lang in sample_langs:
        tmpdir.mkdir(lang)
        tmpdir.mkdir(f"{lang}/links")
        tmpdir.mkdir(f"{lang}/descs_pages")

    # create links which are npz files. Each file contains multiple tokens and the same number of qids.
    for j, lang in enumerate(sample_langs):
        for i in range(int(10**3)):
            example_tokens = np.array([[i, 2], [i, 3], [i, 0]])
            example_qids = np.array([j, j, j])

            np.savez(
                tmpdir / f"{lang}/links/mentions_{i}.npz",
                tokens=example_tokens,
                qids=example_qids,
            )

    # create descriptions which are npz files. Each file contains multiple lines and the same number of qids.
    for j, lang in enumerate(sample_langs):
        for i in range(int(10**3)):
            example_lines = np.array([1, 1, 1, 1])
            # each qid can appear at most once
            start_qid = j * 100000 + i * 1000
            example_qids = np.array(
                [start_qid, start_qid + 1, start_qid + 2, start_qid + 3]
            )

            np.savez(
                tmpdir / f"{lang}/descs_pages/mentions_{i}.npz",
                tokens=example_lines,
                qids=example_qids,
            )

    yield tmpdir


@pytest.fixture
def damuel_paths(setup_teardown):
    return DamuelPaths(str(setup_teardown))


@pytest.fixture
def output_dir(tmpdir):
    tmpdir.mkdir("output")

    yield tmpdir / "output"


class Test_LinksCreator:
    @pytest.fixture
    def links_creator(self, damuel_paths, sample_langs, output_dir):
        return _LinksCreator(damuel_paths, sample_langs, Path(output_dir))

    def test_run(self, links_creator, output_dir):
        # test should
        # check that output path is empty
        # run
        # check that output path is not empty
        # validate that output path contains expected files
        # validate number, keys and content of each file, and the language of each file (which can be inferred from the qid)

        output_dir = Path(output_dir) / "links"

        assert not any(output_dir.iterdir())

        links_creator.run()

        assert any(output_dir.iterdir())

        prev_qids = []
        all_qids = []

        for file in output_dir.iterdir():
            print(output_dir, file)
            data = np.load(file)
            tokens = data["tokens"]
            qids = data["qids"]

            prev_qids.append(qids)
            all_qids.extend(qids)

        # check that files are at least a little bit shuffled
        assert any(
            not np.all(prev_qids[i] == prev_qids[i + 1])
            for i in range(len(prev_qids) - 1)
        )

        assert set(all_qids) == {0, 1, 2}


class Test_KBCreator:
    @pytest.fixture
    def kb_creator(self, damuel_paths, sample_langs, output_dir):
        return _KBCreator(damuel_paths, sample_langs, Path(output_dir))

    def test_run(self, kb_creator, output_dir):
        output_dir = Path(output_dir) / "descs_pages"

        assert not any(output_dir.iterdir())

        kb_creator.run()

        assert any(output_dir.iterdir())

        all_qids = []

        for file in output_dir.iterdir():
            data = np.load(file)
            qids = data["qids"]

            all_qids.extend(qids)
        assert len(set(all_qids)) == len(all_qids)

    def test_get_mapping_from_counts_and_lang_sizes(self, kb_creator):
        qid_lang_counts = {
            1: Counter({"en": 3, "es": 1, "fr": 2}),
            2: Counter({"fr": 5, "en": 4}),
            3: Counter({"es": 1}),
            4: Counter({"fr": 5, "en": 5}),
            5: Counter({"fr": 0}),
        }

        lang_sizes = {
            "en": 10,
            "fr": 8,
            "es": 7,
        }

        result = kb_creator._get_mapping_from_counts_and_lang_sizes(
            qid_lang_counts,
            lang_sizes,
        )

        assert result == {1: ["en"], 2: ["fr"], 3: ["es"], 4: ["en"], 5: ["fr"]}

    def test_run_with_langs_per_qid_2(self, damuel_paths, sample_langs, output_dir):
        kb_creator = _KBCreator(
            damuel_paths, sample_langs, Path(output_dir), langs_per_qid=2
        )
        output_dir = Path(output_dir) / "descs_pages"

        assert not any(output_dir.iterdir())

        kb_creator.run()

        assert any(output_dir.iterdir())

        all_qids = []
        lang_counts = defaultdict(int)

        for file in output_dir.iterdir():
            data = np.load(file)
            qids = data["qids"]
            lang = file.stem.split("_")[1]

            all_qids.extend(qids)
            lang_counts[lang] += len(qids)

        unique_qids = set(all_qids)
        assert len(unique_qids) * 2 >= len(all_qids) > len(unique_qids)

        # Check that we have a mix of languages
        assert len(lang_counts) > 1


class TestMultilingualDatasetCreator:
    @pytest.fixture
    def dataset_creator(self, sample_langs, output_dir, tmp_path):
        return MultilingualDatasetCreator(
            Path(tmp_path), sample_langs, Path(output_dir)
        )

    @patch("multilingual_dataset.creator._KBCreator.run")
    @patch("multilingual_dataset.creator._LinksCreator.run")
    def test_run(self, mock_links_run, mock_kb_run, dataset_creator):
        dataset_creator.run()

        # Check that the appropriate methods were called
        mock_kb_run.assert_called_once()
        mock_links_run.assert_called_once()
