import concurrent.futures
import logging
from collections import Counter, defaultdict
from collections.abc import Iterable
from itertools import zip_longest
from pathlib import Path
from typing import Union

import numpy as np

from multilingual_dataset.mixer import Mixer
from tqdm import tqdm
from utils.damuel_paths import DamuelPaths
from utils.loaders import load_mentions, load_qids

_logger = logging.getLogger("multilingual_dataset.creator")


class _LinksCreator:
    def __init__(
        self, damuel_paths: DamuelPaths, langs: list[str], dest_dir: Path
    ) -> None:
        self.damuel_paths: DamuelPaths = damuel_paths
        self.langs: list[str] = langs
        self.dest_links_dir: Path = dest_dir / "links"
        self.dest_links_dir.mkdir(parents=True, exist_ok=True)

        self.single_mixer = Mixer(buffer_size=1)
        self.standard_mixer = Mixer(buffer_size=200)

    def run(self) -> None:
        """Gathers links from all languages and writes them to dest_dir.

        The resulting files have same structure as the original, each file is a mix of links from all languages.
        """
        link_dir_paths = self.damuel_paths.get_links(self.langs)
        link_file_paths = self._get_link_file_paths(link_dir_paths)

        out_file_paths = []

        for i, file_paths in tqdm(
            enumerate(zip_longest(*link_file_paths)),
            desc="Copying links",
            total=max(len(link_file_paths) for link_file_paths in link_file_paths),
        ):
            out_file_path = self.dest_links_dir / f"mentions_{i}.npz"
            self._copy_files((path for path in file_paths if path), out_file_path)

            out_file_paths.append(out_file_path)

        self.single_mixer.mix(out_file_paths, n_of_mixings=1, compress_output=False)
        self.standard_mixer.mix(out_file_paths, n_of_mixings=5, compress_output=True)

    def _copy_files(
        self, source_file_paths: Iterable[Path], dest_file_path: Path
    ) -> None:
        def load_file(file_path: Path) -> tuple[np.ndarray, np.ndarray]:
            return load_mentions(file_path)

        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = list(executor.map(load_file, source_file_paths))

        tokens, qids = zip(*results)

        tokens = np.concatenate(tokens)
        qids = np.concatenate(qids)

        np.savez(dest_file_path, tokens=tokens, qids=qids)

    def _get_link_file_paths(self, link_dir_paths: list[Path]) -> list[list[Path]]:
        link_file_paths = []
        for link_dir_path in link_dir_paths:
            link_file_paths.append(
                [link_dir_path / file_name for file_name in link_dir_path.iterdir()]
            )
        return link_file_paths


class _KBCreator:
    def __init__(
        self,
        damuel_paths: DamuelPaths,
        langs: list[str],
        dest_dir: Path,
        langs_per_qid: int = 1,
    ) -> None:
        self.damuel_paths: DamuelPaths = damuel_paths
        self.langs: list[str] = langs
        self.dest_descs_dir: Path = dest_dir / "descs_pages"
        self.dest_descs_dir.mkdir(parents=True, exist_ok=True)
        self.dest_dir: Path = dest_dir
        self.langs_per_qid: int = langs_per_qid

    def run(self) -> None:
        qid_lang_mapping = self._get_qid_lang_mapping()
        lang_qid_lists = self._group_qids_by_lang(qid_lang_mapping)
        self._copy_chosen_pages(lang_qid_lists)

    def _copy_chosen_pages(self, lang_qid_lists: dict[str, list[int]]) -> None:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = []
            for lang in self.langs:
                filepaths = self._get_file_paths(self.damuel_paths.get_pages([lang]))
                futures.append(
                    executor.submit(
                        self._copy_chosen_pages_from_lang,
                        lang_qid_lists[lang],
                        filepaths,
                        lang,
                    )
                )

            concurrent.futures.wait(futures)

    def _copy_chosen_pages_from_lang(
        self, wanted_qids: list[int], filepaths: list[Path], lang: str
    ) -> None:
        for i, descs_file_path in enumerate(filepaths):
            tokens, qids = load_mentions(descs_file_path)

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
        for qid, langs in tqdm(
            qid_lang_mapping.items(),
            desc="Grouping QIDs by language",
            total=len(qid_lang_mapping),
        ):
            for lang in langs:
                lang_qid_lists[lang].append(qid)
        return lang_qid_lists

    def _get_qid_lang_mapping(self) -> dict[int, list[str]]:
        qid_lang_counts, lang_sizes = self._get_qid_lang_counts()
        return self._get_mapping_from_counts_and_lang_sizes(qid_lang_counts, lang_sizes)

    def _get_qid_lang_counts(self) -> tuple[dict[int, dict[str, int]], dict[str, int]]:
        qid_lang_counts = self._init_qid_lang_counts()
        lang_sizes = defaultdict(int)
        for lang, dir_filepath in tqdm(
            zip(self.langs, self.damuel_paths.get_links(self.langs)),
            desc="Counting QIDs in links",
            total=len(self.langs),
        ):
            links_filepaths = self._get_file_paths([dir_filepath])
            qids = self._load_qids_from_filepaths(links_filepaths)
            # Why use np.unique here when the sorting is n log n compared to a simple loop?
            # Looping through qids and incrementing the count by 1 is actually slower because
            # of the overhead of dictionary accessing.
            unique_qids, counts = np.unique(qids, return_counts=True)
            for qid, count in zip(unique_qids, counts):
                if lang not in qid_lang_counts[qid]:
                    qid_lang_counts[qid][lang] = 0
                qid_lang_counts[qid][lang] += count
            lang_sizes[lang] += len(qids)

        return qid_lang_counts, lang_sizes

    def _load_qids_from_filepaths(self, filepaths: list[Path]) -> list[int]:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            qids = list(executor.map(load_qids, filepaths))

        return [item for sublist in qids for item in sublist]

    def _get_mapping_from_counts_and_lang_sizes(
        self, qid_lang_counts: dict[int, Counter], lang_sizes: dict[str, int]
    ) -> dict[int, list[str]]:
        """Chooses the most common language for each QID ties are broken in favor of the larger language.

        Args:
            qid_lang_counts (dict[int, Counter])
            lang_sizes (dict[str, int])

        Returns:
            dict[int, list[str]]
        """
        qid_lang_mapping = {}
        for qid, lang_counts in tqdm(
            qid_lang_counts.items(),
            desc="Mapping QIDs to languages",
            total=len(qid_lang_counts),
        ):
            items_by_importance = sorted(
                lang_counts.items(), key=lambda x: (-x[1], -lang_sizes[x[0]])
            )
            n_of_langs_to_choose = min(self.langs_per_qid, len(items_by_importance))
            choosen_langs = [
                item[0] for item in items_by_importance[:n_of_langs_to_choose]
            ]
            qid_lang_mapping[qid] = choosen_langs

        return qid_lang_mapping

    def _init_qid_lang_counts(self) -> dict[int, dict[str, int]]:
        qid_lang_counts = defaultdict(lambda: defaultdict(int))
        for lang in self.langs:
            for descs_file_path in self._get_file_paths(
                self.damuel_paths.get_pages([lang])
            ):
                qids = load_qids(descs_file_path)
                for qid in qids:
                    qid_lang_counts[qid][lang] += 1
        return qid_lang_counts

    def _get_file_paths(self, dir_paths: list[Path]) -> list[Path]:
        return [
            dir_path / file_path.name
            for dir_path in dir_paths
            for file_path in dir_path.iterdir()
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
        _logger.info("Starting to create KB")
        self._kb_creator.run()
        _logger.info("Finished creating KB")

        _logger.info("Starting to create links")
        self._links_creator.run()
        _logger.info("Finished creating links")


def create_multilingual_dataset(
    source_dir: Union[str, Path],
    langs: list[str],
    dest_dir: Union[str, Path],
) -> None:
    MultilingualDatasetCreator(Path(source_dir), langs, Path(dest_dir)).run()


def run_kb_creator(
    source_dir: Union[str, Path],
    langs: list[str],
    dest_dir: Union[str, Path],
    langs_per_qid: int,
) -> None:
    _KBCreator(DamuelPaths(source_dir), langs, Path(dest_dir), langs_per_qid).run()
