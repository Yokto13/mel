import itertools
import logging
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Optional

sys.stdout.reconfigure(line_buffering=True, write_through=True)

from collections.abc import Iterable

import numpy as np
import wandb

from data_processors.index.index import Index

from utils.argument_wrappers import paths_exist
from utils.multifile_dataset import MultiFileDataset

_logger = logging.getLogger("baselines.olpeat.find_recall")


def get_unique_n(iterable: Iterable, n: int):
    seen = set()
    for i in iterable:
        if i not in seen:
            seen.add(i)
            yield i
        if len(seen) == n:
            break


class RecallCalculator:
    def __init__(self, scann_index, qids_in_index) -> None:
        self.scann_index = scann_index
        self.qids_in_index = qids_in_index

    def recall(self, mewsli_embs, mewsli_qids, k: int):
        qid_was_present = self._process_for_recall(mewsli_embs, mewsli_qids, k)
        return self._calculate_recall(qid_was_present)

    def _calculate_recall(self, qid_was_present):
        return sum(qid_was_present) / len(qid_was_present)

    def _get_neighboring_qids(self, queries_embs, k):
        qids_per_query = []
        neighbors, _ = self.scann_index.search_batched(
            queries_embs, final_num_neighbors=max(100000, k)
        )
        for ns in neighbors:
            ns_qids = self.qids_in_index[ns]
            unique_ns_qids = list(get_unique_n(ns_qids, k))
            qids_per_query.append(unique_ns_qids)
        return qids_per_query

    def _process_for_recall(self, mewsli_embs, mewsli_qids, k):
        qid_was_present = []

        for emb, qid in zip(mewsli_embs, mewsli_qids):
            negihboring_qids = self._get_neighboring_qids([emb], k)
            qid_was_present.append(qid in negihboring_qids[0])

        return qid_was_present


class _MewsliLoader:
    def __init__(self, data_path):
        if isinstance(data_path, str):
            data_path = Path(data_path)
        self.data_path = data_path

    def _load_embs(self) -> tuple[np.ndarray, np.ndarray]:
        data = np.load(self.data_path)
        embs_all = data["embs"]
        qids_all = data["qids"]
        return embs_all, qids_all

    def get_data(self) -> tuple[np.ndarray, np.ndarray]:
        _logger.debug("Loading MEWSLI entities...")
        mewsli_embs, mewsli_qids = self._load_embs()
        mewsli_embs = np.array(mewsli_embs)

        mewsli_embs = mewsli_embs / np.linalg.norm(mewsli_embs, axis=1, keepdims=True)

        return mewsli_embs, mewsli_qids


class _DamuelLoader:
    def __init__(
        self,
        descs_embs_path: str,
        links_embs_path: Optional[str],
        damuel_tokens_path: str,
    ) -> None:
        self.descs_embs_path = descs_embs_path
        self.links_embs_path = links_embs_path
        self.damuel_tokens_path = damuel_tokens_path

    def get_data(self, R: int) -> tuple[np.ndarray, np.ndarray]:
        tokens_qids = self._get_wanted_qids_per_tokens(R)
        embs, qids = self._get_data_for_searcher(tokens_qids)
        _logger.debug(f"Damuel embs shape {embs.shape}")
        _logger.debug(f"Damuel qids shape {qids.shape}")
        return embs, qids

    def _construct_tokens_dataset(self) -> MultiFileDataset:
        return MultiFileDataset(self.damuel_tokens_path)

    def _get_wanted_qids_per_tokens(self, R: int) -> dict[int, list[int]]:
        tokens_dataset = self._construct_tokens_dataset()

        def load_data() -> defaultdict[tuple[int], Counter[int]]:
            data = defaultdict(Counter)
            for toks, qid in tokens_dataset:
                data[tuple(toks)][qid] += 1
            return data

        def choose_top_R(tokens_qids: dict[int, Counter[int]]) -> dict[int, list[int]]:
            for k in tokens_qids:
                tokens_qids[k] = [x[0] for x in tokens_qids[k].most_common(R)]
            return tokens_qids

        all_tokens_qids = load_data()
        return choose_top_R(all_tokens_qids)

    def _get_data_for_searcher(
        self,
        tokens_qids: dict[int, list[int]],
    ) -> tuple[np.ndarray, np.ndarray]:
        def get_cnt_of_searcher_items() -> int:
            return sum((len(v) for v in tokens_qids.values()))

        def first(x: Iterable[Any]) -> Any:
            return next(iter(x))

        tokens_embs = self._get_tokens_embs_mapping()
        searcher_items_cnt = get_cnt_of_searcher_items()
        embs = np.empty(
            (searcher_items_cnt, len(first(tokens_embs.values()))), dtype=np.float16
        )
        qids = np.empty(searcher_items_cnt, dtype=np.int32)

        idx = 0
        for toks, item_qids in tokens_qids.items():
            qids_per_token = len(item_qids)
            next_idx = idx + qids_per_token
            emb = tokens_embs[tuple(toks)]
            embs[idx:next_idx] = emb
            qids[idx:next_idx] = item_qids
            idx = next_idx
        return embs, qids

    def _get_tokens_embs_mapping(self) -> dict[tuple[int], np.ndarray]:
        def load(path: Optional[Path] = None) -> tuple[list[Any], list[Any]]:
            if path is None:
                return [], []
            d = np.load(path)
            return d["tokens"], d["embs"]

        mapping = {}
        for toks, embs in itertools.chain(
            zip(*load(self.descs_embs_path)),
            zip(*load(self.links_embs_path)) if self.links_embs_path else [],
        ):
            mapping[tuple(toks)] = embs
        return mapping


class OLPEAT:
    @paths_exist(path_arg_ids=[1, 3, 4])
    def __init__(
        self,
        descs_embs_path: str,
        links_embs_path: str | None,
        mewsli_embs_path: str,
        damuel_tokens_path: str,
    ) -> None:
        """Inits OLPEAT which can be used to evaluate alias tables with embeddings.

        Args:
            descs_embs_path (str): Path to embedded DaMuEL mentions from descriptions.
            links_embs_path (str): Path to embedded DaMuEL mentions from links.
            mewsli_embs_path (str): Path to embedded Mewsli-9 language.
            damuel_tokens_path (str): Path to directory with DaMuEL tokens corresponding to the embeddings above.

        Note:
            All embeddings above should be produced by olpeat.at_embeddings which saves not only embeddings but also corresponding
            tokens. This allows OLPEAT to construct the dataset from the knowledge of damuel_tokens_path.
            Please read README in olpeat folder for more info.
        """
        mewsli_loader = _MewsliLoader(mewsli_embs_path)
        self.mewsli_embs, self.mewsli_qids = mewsli_loader.get_data()

        self._damuel_loader = _DamuelLoader(
            descs_embs_path, links_embs_path, damuel_tokens_path
        )

    def _get_scann_index(self, embs: np.ndarray, qids: np.ndarray) -> "Scann":
        _logger.debug("Building SCANN index...")
        index = Index(embs, qids, default_index_build=False)
        index.build_index(
            num_leaves=5 * int(np.sqrt(len(qids))),
            num_leaves_to_search=800,
            training_sample_size=len(qids),
            reordering_size=1000,
        )
        return index.scann_index

    def find_recall(self, R: int) -> float:
        damuel_embs, damuel_qids = self._damuel_loader.get_data(R)
        _logger.debug(f"len(damuel_embs) {len(damuel_embs)}")

        scann_index = self._get_scann_index(damuel_embs, damuel_qids)
        rc = RecallCalculator(scann_index, damuel_qids)

        _logger.debug("Calculating recall...")
        recall = rc.recall(self.mewsli_embs, self.mewsli_qids, R)
        wandb.log({f"recall_at_{R}": recall})
        _logger.info(f"Recall at {R}: {recall}")
        return recall


def find_recall(descs, tokens, mewsli, R, links=None) -> float:
    olpeat = OLPEAT(descs, links, mewsli, tokens)
    return olpeat.find_recall(R)
