import json
import logging
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

_logger = logging.getLogger("finetunings.evaluation.find_recall")


def load_embs_and_qids_with_normalization(
    path: str | Path,
) -> tuple[np.ndarray, np.ndarray]:
    embs, qids = load_embs_and_qids(path)
    embs = embs / np.linalg.norm(embs, axis=1, keepdims=True)
    return embs, qids


def get_scann_index(embs, qids):
    _logger.info("Building SCANN index...")
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


@paths_exist(path_arg_ids=[0, 1])
def find_recall(
    damuel_entities: str,
    mewsli: str,
    recalls: list[int],
) -> None:
    damuel_embs, damuel_qids = load_embs_and_qids_with_normalization(damuel_entities)
    mewsli_embs, mewsli_qids = load_embs_and_qids_with_normalization(mewsli)

    _logger.info(
        f"Shapes: damuel_embs={damuel_embs.shape}, damuel_qids={damuel_qids.shape}"
    )
    # searcher = get_scann_searcher(damuel_embs, damuel_qids)
    # searcher = get_faiss_searcher(damuel_embs, damuel_qids)
    searcher = get_brute_force_searcher(damuel_embs, damuel_qids)
    rc = RecallCalculator(searcher)

    for R in recalls:
        _logger.info("Calculating recall...")
        recall = rc.recall(mewsli_embs, mewsli_qids, R)
        wandb.log({f"recall_at_{R}": recall})
        _logger.info(f"Recall at {R}: {recall}")


def find_candidates(
    damuel_entities: str, candidates_path: str, mewsli: str, recall: int
) -> None:
    damuel_embs, damuel_qids = load_embs_and_qids_with_normalization(damuel_entities)
    mewsli_embs, mewsli_qids = load_embs_and_qids_with_normalization(mewsli)
    searcher = get_brute_force_searcher(damuel_embs, damuel_qids)
    rc = RecallCalculator(searcher)

    r, candidate_qids = rc.recall(mewsli_embs, mewsli_qids, recall, verbose=True)

    _logger.info(f"Recall at {recall}: {r}")

    np.savez_compressed(candidates_path, candidate_qids=candidate_qids)


if __name__ == "__main__":
    fire.Fire(find_recall)
