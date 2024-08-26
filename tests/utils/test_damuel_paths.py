from pathlib import Path
import pytest

from utils.damuel_paths import DamuelPaths


@pytest.fixture
def damuel_paths(tmpdir):
    tmpdir.mkdir("en")
    tmpdir.mkdir("en/links")
    tmpdir.mkdir("en/descs_pages")
    tmpdir.mkdir("es")
    tmpdir.mkdir("es/links")
    tmpdir.mkdir("es/descs_pages")
    tmpdir.mkdir("de")
    tmpdir.mkdir("de/links")
    tmpdir.mkdir("de/descs_pages")

    return DamuelPaths(tmpdir)


def test_get_links_one_lang(damuel_paths, tmpdir):
    link_paths = damuel_paths.get_links(["en"])

    assert len(link_paths) == 1
    assert isinstance(link_paths[0], Path)
    assert link_paths[0] == Path(tmpdir / "links/en")


def test_get_links(damuel_paths):
    links_paths = damuel_paths.get_links()
    assert len(links_paths) == 3


def test_get_links_two(damuel_paths):
    links_paths = damuel_paths.get_links(["es", "en"])
    assert len(links_paths) == 2


def test_get_links_invalid(damuel_paths):
    with pytest.raises(ValueError):
        links_paths = damuel_paths.get_links(["fa", "en"])


def test_get_pages_one_lang(damuel_paths, tmpdir):
    pages_paths = damuel_paths.get_pages(["en"])

    assert len(pages_paths) == 1
    assert isinstance(pages_paths[0], Path)
    assert pages_paths[0] == Path(tmpdir / "descs_pages/en")


def test_get_pages(damuel_paths):
    pages_paths = damuel_paths.get_pages()
    assert len(pages_paths) == 3


def test_get_pages_two(damuel_paths):
    pages_paths = damuel_paths.get_pages(["es", "en"])
    assert len(pages_paths) == 2


def test_get_pages_invalid(damuel_paths):
    with pytest.raises(ValueError):
        damuel_paths.get_pages(["fa", "en"])
