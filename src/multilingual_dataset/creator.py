from collections import Counter, defaultdict
from itertools import zip_longest
from pathlib import Path
from typing import Union

import numpy as np

from multilingual_dataset.mixer import Mixer
from utils.damuel_paths import DamuelPaths
from utils.loaders import load_mentions


class _LinksCreator:
    def __init__(
        self, damuel_paths: DamuelPaths, langs: list[str], dest_dir: Path
    ) -> None:
        self.damuel_paths: DamuelPaths = damuel_paths
        self.langs: list[str] = langs
        self.dest_links_dir: Path = dest_dir / "links"
        self.dest_links_dir.mkdir(parents=True, exist_ok=True)

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
            out_file_path = self.dest_links_dir / f"mentions_{i}.npz"
            self._copy_files([path for path in file_paths if path], out_file_path)

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
        self, damuel_paths: DamuelPaths, langs: list[str], dest_dir: Path
    ) -> None:
        self.damuel_paths: DamuelPaths = damuel_paths
        self.langs: list[str] = langs
        self.dest_descs_dir: Path = dest_dir / "descs_pages"
        self.dest_descs_dir.mkdir(parents=True, exist_ok=True)
        self.dest_dir: Path = dest_dir

    def run(self) -> None:
        qid_lang_mapping = self._get_qid_lang_mapping()
        lang_qid_lists = self._group_qids_by_lang(qid_lang_mapping)
        self._copy_chosen_pages(lang_qid_lists)

    def _copy_chosen_pages(self, lang_qid_lists: dict[str, list[int]]) -> None:
        for lang in self.langs:
            filepaths = self._get_file_paths(self.damuel_paths.get_pages([lang]))
            self._copy_chosen_pages_from_lang(lang_qid_lists[lang], filepaths, lang)

    def _copy_chosen_pages_from_lang(
        self, wanted_qids: list[int], filepaths: list[Path], lang: str
    ) -> None:
        for i, descs_file_path in enumerate(filepaths):
            d = np.load(descs_file_path)
            tokens = d["tokens"]
            qids = d["qids"]

            index = np.isin(qids, list(wanted_qids))
            chosen_tokens = tokens[index]
            chosen_qids = qids[index]

            if len(chosen_tokens) == 0:
                continue

            np.savez_compressed(
                self.dest_descs_dir / f"mentions_{lang}_{i}.npz",
                tokens=chosen_tokens,
                qids=chosen_qids,
            )

    def _group_qids_by_lang(
        self, qid_lang_mapping: dict[int, str]
    ) -> dict[str, list[int]]:
        lang_qid_lists = defaultdict(list)
        for qid, lang in qid_lang_mapping.items():
            lang_qid_lists[lang].append(qid)
        return lang_qid_lists

    def _get_qid_lang_mapping(self) -> dict[int, str]:
        qid_lang_counts, lang_sizes = self._get_qid_lang_counts()
        return self._get_mapping_from_counts_and_lang_sizes(qid_lang_counts, lang_sizes)

    def _get_qid_lang_counts(self) -> tuple[dict[int, int], dict[str, int]]:
        qid_lang_counts = self._init_qid_lang_counts()
        lang_sizes = defaultdict(int)
        for lang in self.langs:
            for link_file_path in self._get_file_paths(
                self.damuel_paths.get_links([lang])
            ):
                qids = np.load(link_file_path)["qids"]

                for qid in qids:
                    if lang not in qid_lang_counts[qid]:
                        qid_lang_counts[qid][lang] = 0
                    qid_lang_counts[qid][lang] += 1
            lang_sizes[lang] += len(qids)

        return qid_lang_counts, lang_sizes

    def _get_mapping_from_counts_and_lang_sizes(
        self, qid_lang_counts: dict[int, Counter], lang_sizes: dict[str, int]
    ) -> dict[int, str]:
        """Chooses the most common language for each QID ties are broken in favor of the larger language.

        Args:
            qid_lang_counts (dict[int, Counter])
            lang_sizes (dict[str, int])

        Returns:
            dict[int, str]
        """
        qid_lang_mapping = {}
        for qid, lang_counts in qid_lang_counts.items():
            ordered_lang_counts = sorted(
                lang_counts.items(),
                key=lambda x: (x[1], lang_sizes[x[0]]),
                reverse=True,
            )
            qid_lang_mapping[qid] = ordered_lang_counts[0][0]

        return qid_lang_mapping

    def _init_qid_lang_counts(self) -> dict[int, dict[str, int]]:
        qid_lang_counts = defaultdict(lambda: defaultdict(int))
        for lang in self.langs:
            for descs_file_path in self._get_file_paths(
                self.damuel_paths.get_pages([lang])
            ):
                qids = np.load(descs_file_path)["qids"]
                for qid in qids:
                    qid_lang_counts[qid][lang] += 1
        return qid_lang_counts

    def _get_file_paths(self, dir_paths: list[Path]) -> list[Path]:
        return [
            dir_path / file_name
            for dir_path in dir_paths
            for file_name in dir_path.iterdir()
        ]


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


def create_multilingual_dataset(
    source_dir: Union[str, Path],
    langs: list[str],
    dest_dir: Union[str, Path],
) -> None:
    MultilingualDatasetCreator(Path(source_dir), langs, Path(dest_dir)).run()
