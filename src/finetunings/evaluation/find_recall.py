from pathlib import Path
from utils.loaders import load_embs_and_qids

import fire
import numpy as np
import wandb

from data_processors.index.index import Index
from utils.argument_wrappers import paths_exist
from models.recall_calculator import RecallCalculator
from models.searchers.scann_searcher import ScaNNSearcher


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


@paths_exist(path_arg_ids=[0, 1])
def find_recall(
    damuel_entities: str,
    mewsli: str,
    recalls: list[int],
) -> None:
    damuel_embs, damuel_qids = load_embs_and_qids_with_normalization(damuel_entities)

    mewsli_embs, mewsli_qids = load_embs_and_qids_with_normalization(mewsli)

    print(damuel_embs.shape, damuel_qids.shape)
    searcher = get_scann_searcher(damuel_embs, damuel_qids)
    rc = RecallCalculator(searcher)

    for R in recalls:
        print("Calculating recall...")
        recall = rc.recall(mewsli_embs, mewsli_qids, R)
        wandb.log({f"recall_at_{R}": recall})
        print(f"Recall at {R}:", recall)


if __name__ == "__main__":
    fire.Fire(find_recall)
