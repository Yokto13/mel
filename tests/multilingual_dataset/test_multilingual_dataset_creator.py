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
        for i in range(10000):
            example_lines = np.array([1, 1, 1, 1])
            example_qids = np.array([j, j, j, j])

            np.savez(
                tmpdir / f"{lang}/descs_pages/mentions_{i}.npz",
                toknes=example_lines,
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

        prev_qids = None
        all_qids = []

        for file in output_dir.iterdir():
            print(output_dir, file)
            data = np.load(file)
            tokens = data["tokens"]
            qids = data["qids"]

            if prev_qids is not None:
                # run should shuffle
                # all link files in the dummy dataset have same qids
                # without shuffling, qids would be the same between files
                assert len(qids) != len(prev_qids) or not np.all(qids == prev_qids)

            prev_qids = qids
            all_qids.extend(qids)

        assert set(all_qids) == {0, 1, 2}


class Test_KBCreator:
    @pytest.fixture
    def kb_creator(self, damuel_paths, sample_langs, output_dir):
        return _KBCreator(damuel_paths, sample_langs, output_dir)

    def test_run(self, kb_creator, output_dir):
        # Set up any necessary directory structure in tmp_path
        # ...

        kb_creator.run()

        # Add assertions to check the expected outcomes
        # For example:
        # assert (tmp_path / 'expected_file.txt').exists()
        # assert some_condition_is_met


class TestMultilingualDatasetCreator:
    @pytest.fixture
    def dataset_creator(self, sample_langs, output_dir, tmp_path):
        return MultilingualDatasetCreator(tmp_path, sample_langs, output_dir)

    @patch("multilingual_dataset.creator._KBCreator.run")
    @patch("multilingual_dataset.creator._LinksCreator.run")
    def test_run(self, mock_links_run, mock_kb_run, dataset_creator):
        dataset_creator.run()

        # Check that the appropriate methods were called
        mock_kb_run.assert_called_once()
        mock_links_run.assert_called_once()
