import os
from pathlib import Path
from unittest.mock import patch

import pytest
from utils.damuel_paths import all_paths_must_exist, DamuelPaths, path_must_exist


def mock_path_exists_enforcer(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Path does not exist: {path}")


@pytest.fixture
def setup_teardown(tmpdir):
    # Setup
    for lang in ["en", "es", "de"]:
        tmpdir.mkdir(lang)
        tmpdir.mkdir(f"{lang}/links")
        tmpdir.mkdir(f"{lang}/descs_pages")

    yield tmpdir


@pytest.fixture
def damuel_paths(setup_teardown):
    return DamuelPaths(str(setup_teardown))


def test_get_links_one_lang(damuel_paths, setup_teardown):
    link_paths = damuel_paths.get_links(["en"])
    assert len(link_paths) == 1
    assert isinstance(link_paths[0], Path)
    assert link_paths[0] == Path(setup_teardown) / "en/links"


def test_get_links(damuel_paths):
    links_paths = damuel_paths.get_links()
    assert len(links_paths) == 3


def test_get_links_two(damuel_paths):
    links_paths = damuel_paths.get_links(["es", "en"])
    assert len(links_paths) == 2


def test_get_links_invalid(damuel_paths):
    with pytest.raises(FileNotFoundError):
        links_paths = damuel_paths.get_links(["fa", "en"])


def test_get_pages_one_lang(damuel_paths, setup_teardown):
    pages_paths = damuel_paths.get_pages(["en"])
    assert len(pages_paths) == 1
    assert isinstance(pages_paths[0], Path)
    assert pages_paths[0] == Path(setup_teardown) / "en/descs_pages"


def test_get_pages(damuel_paths):
    pages_paths = damuel_paths.get_pages()
    assert len(pages_paths) == 3


def test_get_pages_two(damuel_paths):
    pages_paths = damuel_paths.get_pages(["es", "en"])
    assert len(pages_paths) == 2


def test_get_pages_invalid(damuel_paths):
    with pytest.raises(FileNotFoundError):
        damuel_paths.get_pages(["fa", "en"])


def test_get_language_missing(damuel_paths):
    with pytest.raises(FileNotFoundError):
        damuel_paths.get_language("fa")


def test_get_language(damuel_paths, setup_teardown):
    lang_path = damuel_paths.get_language("en")
    assert lang_path == Path(setup_teardown) / "en"


def test_root_damuel_dir_type(damuel_paths):
    assert isinstance(damuel_paths.root_damuel_dir, Path)


def test_path_must_exist_decorator(setup_teardown):
    @path_must_exist
    def get_existing_file():
        return str(setup_teardown / "en")

    @path_must_exist
    def get_non_existing_file():
        return str(setup_teardown / "fa")

    # Test with existing file
    with patch(
        "utils.damuel_paths._path_exists_enforcer",
        side_effect=mock_path_exists_enforcer,
    ):
        assert get_existing_file() == str(setup_teardown / "en")

    # Test with non-existing file
    with patch(
        "utils.damuel_paths._path_exists_enforcer",
        side_effect=mock_path_exists_enforcer,
    ):
        with pytest.raises(FileNotFoundError):
            get_non_existing_file()


def test_all_paths_must_exist_decorator(setup_teardown):
    @all_paths_must_exist
    def get_existing_files():
        return [str(setup_teardown / "en"), str(setup_teardown / "es")]

    @all_paths_must_exist
    def get_mixed_files():
        return [str(setup_teardown / "en"), str(setup_teardown / "non_existent.txt")]

    # Test with all existing files
    with patch(
        "utils.damuel_paths._path_exists_enforcer",
        side_effect=mock_path_exists_enforcer,
    ):
        assert get_existing_files() == [
            str(setup_teardown / "en"),
            str(setup_teardown / "es"),
        ]

    # Test with one non-existing file
    with patch(
        "utils.damuel_paths._path_exists_enforcer",
        side_effect=mock_path_exists_enforcer,
    ):
        with pytest.raises(FileNotFoundError):
            get_mixed_files()
