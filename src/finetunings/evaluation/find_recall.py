from collections import Counter, defaultdict
from hashlib import sha1
from pathlib import Path
import sys

from utils.loaders import load_embs_and_qids

sys.stdout.reconfigure(line_buffering=True, write_through=True)

import fire
import numpy as np
import wandb

from data_processors.index.index import Index
from utils.argument_wrappers import paths_exist


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


def _get_items_count_and_dim(dir_path):
    cnt = 0
    dim = 0
    print("Counting items...")
    print(sorted(dir_path.iterdir()))
    for fname in sorted(dir_path.iterdir()):
        if not fname.name.startswith("embs"):
            continue
        print("Reading", fname)
        if dim == 0:
            embs = np.load(fname)
            dim = embs.shape[1]
        qids = np.load(dir_path / f"qids_{fname.stem.split('_')[1]}.npy")

        for _ in qids:
            cnt += 1
    return cnt, dim


def load_embs(dir_path):
    if isinstance(dir_path, str):
        dir_path = Path(dir_path)

    cnt, dim = _get_items_count_and_dim(dir_path)
    print("Total items:", cnt, "Dimension:", dim)
    embs_all = np.empty((cnt, dim), dtype=np.float32)
    qids_all = np.empty(cnt, dtype=np.int64)

    idx = 0

    for fname in sorted(dir_path.iterdir()):
        if not fname.name.startswith("embs"):
            continue
        print("Loading", fname)
        embs = np.load(fname)
        qids = np.load(dir_path / f"qids_{fname.stem.split('_')[1]}.npy")

        for emb, qid in zip(embs, qids):
            embs_all[idx] = emb
            qids_all[idx] = qid
            idx += 1
    return embs_all, qids_all


def load_damuel(damuel):
    damuel_embs, damuel_qids = load_embs_and_qids(damuel)

    damuel_embs = damuel_embs / np.linalg.norm(damuel_embs, axis=1, keepdims=True)
    return damuel_embs, damuel_qids


def load_mewsli(mewsli):
    print("Loading MEWSLI entities...")
    mewsli_embs, mewsli_qids = load_embs_and_qids(mewsli)

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


@paths_exist(path_arg_ids=[0, 1])
def find_recall(
    damuel_entities: str,
    mewsli: str,
    R,
):
    damuel_embs, damuel_qids = load_damuel(damuel_entities)
    R = min(R, len(damuel_qids))

    mewsli_embs, mewsli_qids = load_mewsli(mewsli)

    print(damuel_embs.shape, damuel_qids.shape)
    scann_index = get_scann_index(damuel_embs, damuel_qids)
    rc = RecallCalculator(scann_index, damuel_qids)

    print("Calculating recall...")
    recall = rc.recall(mewsli_embs, mewsli_qids, R)
    wandb.log({f"recall_at_{R}": recall})
    print(f"Recall at {R}:", recall)


if __name__ == "__main__":
    fire.Fire(find_recall)
