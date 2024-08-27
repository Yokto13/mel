"""
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
def mock_damuel_paths():
    return Mock(spec=DamuelPaths)


class Test_LinksCreator:
    @pytest.fixture
    def links_creator(self, mock_damuel_paths, sample_langs, tmp_path):
        return _LinksCreator(mock_damuel_paths, sample_langs, tmp_path)

    def test_run(self, links_creator, tmp_path):
        # Set up any necessary directory structure in tmp_path
        # ...

        links_creator.run()

        # Add assertions to check the expected outcomes
        # For example:
        # assert (tmp_path / 'expected_file.txt').exists()
        # assert some_condition_is_met


class Test_KBCreator:
    @pytest.fixture
    def kb_creator(self, mock_damuel_paths, sample_langs, tmp_path):
        return _KBCreator(mock_damuel_paths, sample_langs, tmp_path)

    def test_run(self, kb_creator, tmp_path):
        # Set up any necessary directory structure in tmp_path
        # ...

        kb_creator.run()

        # Add assertions to check the expected outcomes
        # For example:
        # assert (tmp_path / 'expected_file.txt').exists()
        # assert some_condition_is_met


class TestMultilingualDatasetCreator:
    @pytest.fixture
    def dataset_creator(self, sample_langs, tmp_path):
        return MultilingualDatasetCreator(tmp_path, sample_langs, tmp_path)

    @patch("your_module._KBCreator.run")
    @patch("your_module._LinksCreator.run")
    def test_run(self, mock_links_run, mock_kb_run, dataset_creator):
        dataset_creator.run()

        # Check that the appropriate methods were called
        mock_kb_run.assert_called_once()
        mock_links_run.assert_called_once()

"""
