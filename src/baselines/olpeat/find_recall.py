from collections import Counter, defaultdict, dataclass
from copy import deepcopy
from hashlib import sha1
import itertools
from pathlib import Path
import sys
from typing import Optional

sys.stdout.reconfigure(line_buffering=True, write_through=True)

import fire
import numpy as np
from torch.utils.data import IterableDataset
import wandb

from data_processors.index.index import Index
from utils.multifile_dataset import MultiFileDataset

# from models.index import Index
from utils.argument_wrappers import paths_exist


class _DamuelPaths:
    descs: Optional[Path]
    links: Optional[Path]


def get_unique_n(iterable, n):
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
        neighbors, dists = self.scann_index.search_batched(
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


def load_embs(dir_path):
    if isinstance(dir_path, str):
        dir_path = Path(dir_path)

    data = np.load(dir_path / "embs_qids.npz")
    embs_all = data["embs"]
    qids_all = data["qids"]
    return embs_all, qids_all


def load_damuel(damuel_entities, damuel_links):
    if damuel_links is not None:
        print("Loading DAMUEL links...")
        damuel_embs, damuel_qids = load_embs(damuel_links)

        print("Loading DAMUEL entities...")
        damuel_embs_entities, damuel_qids_entities = load_embs(damuel_entities)

        damuel_qids = np.concatenate([damuel_qids, damuel_qids_entities])
        del damuel_qids_entities
        damuel_embs = np.concatenate([damuel_embs, damuel_embs_entities])
        del damuel_embs_entities

    else:
        print("Loading DAMUEL entities...")
        damuel_embs, damuel_qids = load_embs(damuel_entities)

    damuel_embs = damuel_embs / np.linalg.norm(damuel_embs, axis=1, keepdims=True)
    return damuel_embs, damuel_qids


def load_mewsli(mewsli):
    print("Loading MEWSLI entities...")
    mewsli_embs, mewsli_qids = load_embs(mewsli)
    mewsli_embs = np.array(mewsli_embs)

    mewsli_embs = mewsli_embs / np.linalg.norm(mewsli_embs, axis=1, keepdims=True)

    return mewsli_embs, mewsli_qids


def get_scann_index(embs, qids):
    print("Building SCANN index...")
    index = Index(embs, qids, default_index_build=False)
    index.build_index(
        num_leaves=5 * int(np.sqrt(len(qids))),
        num_leaves_to_search=800,
        training_sample_size=len(qids),
        reordering_size=1000,
    )
    return index.scann_index


def _get_wanted_qids_per_tokens(tokens_dataset, R):
    def load_data():
        data = defaultdict(Counter)
        for toks, qid in tokens_dataset:
            data[tuple(toks)][qid] += 1
        return data

    def choose_top_R(tokens_qids: dict[int, Counter]):
        for k in tokens_qids:
            tokens_qids[k] = tokens_qids[k].most_common(R)

    all_tokens_qids = load_data()
    return choose_top_R(all_tokens_qids)


def _get_tokens_embs_mapping(damuel_paths: _DamuelPaths):
    def load(path: Optional[Path] = None):
        if path is None:
            return [], []
        d = np.load(path)
        return d["tokens"], d["embs"]

    mapping = {}
    for toks, embs in itertools.chain(
        zip(load(damuel_paths.descs)), zip(load(damuel_paths.links))
    ):
        mapping[tuple[toks]] = embs
    return mapping


def _get_data_for_searcher(tokens_qids: dict, damue_paths: _DamuelPaths):
    def get_cnt_of_searcher_items():
        return sum((len(v) for v in tokens_qids.values()))

    def first(x):
        return next(iter(x))

    tokens_embs = _get_data(damue_paths)
    searcher_items_cnt = get_cnt_of_searcher_items()
    embs = np.array(
        (searcher_items_cnt, len(first(tokens_embs.values()))), dtype=np.float16
    )
    qids = np.array(searcher_items_cnt, dtype=np.int32)

    idx = 0
    for toks, item_qids in tokens_qids.items():
        emb = tokens_embs[tuple(toks)]
        embs[idx : idx + len(qids)] = emb
        qids[idx : idx + len(qids)] = item_qids
    return embs, qids


def _get_data(tokens_dataset: IterableDataset, damuel_paths: _DamuelPaths, R):
    tokens_qids = _get_wanted_qids_per_tokens(deepcopy(tokens_dataset), R)
    embs, qids = _get_data_for_searcher(tokens_qids, damuel_paths)
    return embs, qids


@paths_exist(path_arg_ids=[0, 1, 2])
def find_recall(
    damuel_embs_descs: str,
    damuel_tokens: str,
    mewsli_embs: str,
    R,
    damuel_embs_links: str = None,
):
    dataset = MultiFileDataset(damuel_tokens)
    damuel_embs, damuel_qids = _get_data(
        dataset, _DamuelPaths(damuel_embs_descs, damuel_embs_links), R
    )
    print("len(damuel_embs)", len(damuel_embs))

    mewsli_embs, mewsli_qids = load_mewsli(mewsli_embs)

    print(damuel_embs.shape, damuel_qids.shape)
    scann_index = get_scann_index(damuel_embs, damuel_qids)
    rc = RecallCalculator(scann_index, damuel_qids)

    print("Calculating recall...")
    recall = rc.recall(mewsli_embs, mewsli_qids, R)
    wandb.log({f"recall_at_{R}": recall})
    print(f"Recall at {R}:", recall)


if __name__ == "__main__":
    fire.Fire(find_recall)
