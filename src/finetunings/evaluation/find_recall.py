import json
from pathlib import Path

import fire
import numpy as np
import wandb

from data_processors.index.index import Index
from models.recall_calculator import RecallCalculator
from models.searchers.brute_force_searcher import BruteForceSearcher
from models.searchers.faiss_searcher import FaissSearcher
from models.searchers.scann_searcher import ScaNNSearcher
from utils.argument_wrappers import paths_exist
from utils.loaders import load_embs_and_qids


def load_embs_and_qids_with_normalization(
    path: str | Path,
) -> tuple[np.ndarray, np.ndarray]:
    embs, qids = load_embs_and_qids(path)
    embs = embs / np.linalg.norm(embs, axis=1, keepdims=True)
    return embs, qids


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


def get_scann_searcher(embs, qids) -> ScaNNSearcher:
    searcher = ScaNNSearcher(embs, qids, run_build_from_init=False)
    searcher.build_index(
        num_leaves=5 * int(np.sqrt(len(qids))),
        num_leaves_to_search=800,
        training_sample_size=len(qids),
        reordering_size=1000,
    )
    return searcher


def get_brute_force_searcher(embs, qids) -> BruteForceSearcher:
    searcher = BruteForceSearcher(embs, qids)
    return searcher


def get_faiss_searcher(embs, qids) -> ScaNNSearcher:
    searcher = FaissSearcher(embs, qids)
    return searcher


def _load_json_file(filepath: str | Path) -> dict:
    with open(filepath, "r") as f:
        return json.load(f)


def _convert_qid_keys_to_int(qid_map: dict) -> dict[int, int]:
    return {int(k[1:]): int(v[1:]) for k, v in qid_map.items()}


def load_qids_remap(filepath: str | Path) -> dict[int, int]:
    qid_map = _load_json_file(filepath)
    return _convert_qid_keys_to_int(qid_map)


def qids_remap(qids: np.array, old_to_new_qids_path: str | Path):
    old_to_new_qids = load_qids_remap(old_to_new_qids_path)
    return np.array(
        [q if q not in old_to_new_qids else old_to_new_qids[q] for q in qids]
    )


@paths_exist(path_arg_ids=[0, 1])
def find_recall(
    damuel_entities: str,
    mewsli: str,
    recalls: list[int],
    old_to_new_qids_path: (
        str | Path | None
    ) = "/net/projects/damuel/dev/damuel_1.1-dev_qid_redirects.json",
) -> None:
    damuel_embs, damuel_qids = load_embs_and_qids_with_normalization(damuel_entities)
    mewsli_embs, mewsli_qids = load_embs_and_qids_with_normalization(mewsli)

    if old_to_new_qids_path is not None:
        damuel_qids = qids_remap(damuel_qids, old_to_new_qids_path)
        mewsli_qids = qids_remap(mewsli_qids, old_to_new_qids_path)

    print(damuel_embs.shape, damuel_qids.shape)
    # searcher = get_scann_searcher(damuel_embs, damuel_qids)
    # searcher = get_faiss_searcher(damuel_embs, damuel_qids)
    searcher = get_brute_force_searcher(damuel_embs, damuel_qids)
    rc = RecallCalculator(searcher)

    for R in recalls:
        print("Calculating recall...")
        recall = rc.recall(mewsli_embs, mewsli_qids, R)
        wandb.log({f"recall_at_{R}": recall})
        print(f"Recall at {R}:", recall)


if __name__ == "__main__":
    fire.Fire(find_recall)
