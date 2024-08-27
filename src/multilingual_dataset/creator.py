from copy import deepcopy
from itertools import zip_longest
from pathlib import Path
from typing import Union

import numpy as np

from multilingual_dataset.mixer import Mixer
from utils.damuel_paths import DamuelPaths
from utils.loaders import load_mentions


class _LinksCreator:
    def __init__(
        self, damuel_paths: DamuelPaths, langs: list[str], dest_dir: Union[str, Path]
    ) -> None:
        self.damuel_paths: DamuelPaths = damuel_paths
        self.langs: list[str] = langs
        self.dest_dir: Union[str, Path] = dest_dir / "links"
        self.dest_dir.mkdir(parents=True, exist_ok=True)

        self.single_mixer = Mixer(buffer_size=1)
        self.standard_mixer = Mixer(buffer_size=20)

    def run(self) -> None:
        """Gathers links from all languages and writes them to dest_dir.

        The resulting files have same structure as the original, each file is a mix of links from all languages.
        """
        link_dir_paths = self.damuel_paths.get_links(self.langs)
        link_file_paths = self._get_link_file_paths(link_dir_paths)

        out_file_paths = []

        for i, file_paths in enumerate(zip_longest(*link_file_paths)):
            out_file_path = self.dest_dir / f"mentions_{i}.npz"
            self._copy_files(file_paths, out_file_path)

            out_file_paths.append(out_file_path)

        self.single_mixer.mix(out_file_paths, n_of_mixings=1)
        self.standard_mixer.mix(out_file_paths, n_of_mixings=5)

    def _copy_files(self, source_file_paths: list[Path], dest_file_path: Path) -> None:
        tokens = []
        qids = []
        for source_file_path in source_file_paths:
            tokens_, qids_ = load_mentions(source_file_path)
            tokens.append(tokens_)
            qids.append(qids_)

        tokens = np.concatenate(tokens)
        qids = np.concatenate(qids)

        np.savez_compressed(dest_file_path, tokens=tokens, qids=qids)

    def _get_link_file_paths(self, link_dir_paths: list[Path]) -> list[list[Path]]:
        link_file_paths = []
        for link_dir_path in link_dir_paths:
            link_file_paths.append(
                [link_dir_path / file_name for file_name in link_dir_path.iterdir()]
            )
        return link_file_paths


class _KBCreator:
    def __init__(
        self, damuel_paths: DamuelPaths, langs: list[str], dest_dir: Union[str, Path]
    ) -> None:
        self.damuel_paths: DamuelPaths = damuel_paths
        self.langs: list[str] = langs
        self.dest_dir: Union[str, Path] = dest_dir

    def run(self) -> None:
        qid_lang_mapping = ...


class MultilingualDatasetCreator:
    def __init__(
        self, source_dir: Union[str, Path], langs: list[str], dest_dir: Union[str, Path]
    ) -> None:
        self._damuel_paths: DamuelPaths = DamuelPaths(source_dir)
        self._kb_creator: _KBCreator = _KBCreator(self._damuel_paths, langs, dest_dir)
        self._links_creator: _LinksCreator = _LinksCreator(
            self._damuel_paths, langs, dest_dir
        )

    def run(self) -> None:
        self._kb_creator.run()
        self._links_creator.run()
