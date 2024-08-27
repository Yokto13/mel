from itertools import zip_longest
from pathlib import Path
from typing import Union

from utils.damuel_paths import DamuelPaths

class _Gather:
    def __init__(self, dir):
        self.dir = dir
    
    def run(self):
        paths = self._get_filepath_lists():
        for file_paths in filter(lambda x: x is not None, zip_longest(paths)):
            self._concat_and_shuffle(file_paths)

    def _get_data_dirs(self) -> list[str]:
        pass

    def _get_filepaths_from_dir(self, lang_dir) -> list[str]:
        pass

    def _get_filepath_lists(self) -> list[list[str]]:
        pass

    def _concat_and_shuffle(file_paths):
        pass


class _LinksCreator:
    def __init__(self, damuel_paths: DamuelPaths, langs: list[str], dest_dir: Union[str, Path]) -> None:
        self.damuel_paths: DamuelPaths = damuel_paths
        self.langs: list[str] = langs
        self.dest_dir: Union[str, Path] = dest_dir

    def run(self) -> None:
        pass

class _KBCreator:
    def __init__(self, damuel_paths: DamuelPaths, langs: list[str], dest_dir: Union[str, Path]) -> None:
        self.damuel_paths: DamuelPaths = damuel_paths
        self.langs: list[str] = langs
        self.dest_dir: Union[str, Path] = dest_dir

    def run(self) -> None:
        qid_lang_mapping = ...

class MultilingualDatasetCreator:
    def __init__(self, source_dir: Union[str, Path], langs: list[str], dest_dir: Union[str, Path]) -> None:
        self._damuel_paths: DamuelPaths = DamuelPaths(source_dir)
        self._kb_creator: _KBCreator = _KBCreator(self._damuel_paths, langs, dest_dir)
        self._links_creator: _LinksCreator = _LinksCreator(self._damuel_paths, langs, dest_dir)

    def run(self) -> None:
        self._kb_creator.run()
        self._links_creator.run()