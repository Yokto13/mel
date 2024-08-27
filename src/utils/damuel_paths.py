import functools
import os
from pathlib import Path
from typing import Optional


def _path_exists_enforcer(p: Path):
    if not p.exists():
        raise FileNotFoundError(f"{p} does not exist.")


def path_must_exist(wrapped):
    @functools.wraps(wrapped)
    def _wrapper(*args, **kwargs):
        p = wrapped(*args, **kwargs)
        _path_exists_enforcer(p)
        return p

    return _wrapper


def all_paths_must_exist(wrapped):
    @functools.wraps(wrapped)
    def _wrapper(*args, **kwargs):
        lp = wrapped(*args, **kwargs)
        for p in lp:
            _path_exists_enforcer(p)
        return lp

    return _wrapper


class DamuelPaths:
    def __init__(self, root_damuel_dir: str | Path) -> None:
        self._root_damuel_dir = root_damuel_dir
        if isinstance(self._root_damuel_dir, str):
            self._root_damuel_dir = Path(self._root_damuel_dir)

        self._pages_suffix = "descs_pages"
        self._links_suffix = "links"

    @all_paths_must_exist
    def get_links(self, langs: Optional[list[str]] = None) -> list[Path]:
        langs = self._fill_with_all_if_none(langs)
        return [self.get_language(lang) / self._links_suffix for lang in langs]

    @all_paths_must_exist
    def get_pages(self, langs: Optional[list[str]] = None) -> list[Path]:
        langs = self._fill_with_all_if_none(langs)
        return [self.get_language(lang) / self._pages_suffix for lang in langs]

    @path_must_exist
    def get_language(self, lang) -> Path:
        return self._root_damuel_dir / lang

    @property
    def root_damuel_dir(self) -> Path:
        return self._root_damuel_dir

    def _fill_with_all_if_none(self, langs: list[str] | None) -> list[str]:
        if langs is not None:
            return langs
        return [
            x
            for x in os.listdir(self.root_damuel_dir)
            if self._is_lang_dir_name(x) and (self.root_damuel_dir / x).is_dir()
        ]

    def _is_lang_dir_name(self, candidate_name: str) -> bool:
        """Crude check that candidate_name could be a name of Damuel specific lang directory."""
        return len(candidate_name) == 2
