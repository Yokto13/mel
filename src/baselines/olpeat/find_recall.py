from collections import Counter, defaultdict
from hashlib import sha1
from pathlib import Path
import sys

sys.stdout.reconfigure(line_buffering=True, write_through=True)

import fire
import numpy as np
import wandb

from data_processors.index.index import Index

# from models.index import Index
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


def filter_repeated_embs(damuel_embs, damuel_qids, R):
    """Removes embeddings that are the same.

    Very optional thing, saves some memory, especially when mentions without context are used.
    """
    print("FILTERING EMBS!")
    if not damuel_embs.flags["C_CONTIGUOUS"]:  # need this for sha1 to work
        damuel_embs = np.ascontiguousarray(damuel_embs)

    emb_qid_d = defaultdict(Counter)
    for emb, qid in zip(damuel_embs, damuel_qids):
        emb_qid_d[sha1(emb.tobytes()).hexdigest()][qid] += 1

    # keep only top R qids per emb
    for emb_hash, qid_counter in emb_qid_d.items():
        emb_qid_d[emb_hash] = [qid for qid, _ in qid_counter.most_common(R)]

    new_embs, new_qids = [], []
    for emb, qid in zip(damuel_embs, damuel_qids):
        emb_hash = sha1(emb.tobytes()).hexdigest()
        if qid in emb_qid_d[emb_hash]:
            new_embs.append(emb)
            new_qids.append(qid)

    damuel_embs = np.array(new_embs)
    damuel_qids = np.array(new_qids)

    return damuel_embs, damuel_qids


@paths_exist(path_arg_ids=[0, 1])
def find_recall(
    damuel_entities: str,
    mewsli: str,
    R,
    damuel_links: str = None,
):
    damuel_embs, damuel_qids = load_damuel(damuel_entities, damuel_links)
    R = min(R, len(damuel_qids))
    print("len(damuel_embs)", len(damuel_embs))
    damuel_embs, damuel_qids = filter_repeated_embs(damuel_embs, damuel_qids, R)

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
