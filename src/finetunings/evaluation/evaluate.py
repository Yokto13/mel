import logging

from finetunings.evaluation.find_recall import (
    find_recall_with_searcher, load_embs_and_qids_with_normalization)
from models.searchers.brute_force_searcher import BruteForceSearcher

_RECALLS = [1, 10, 100]

_logger = logging.getLogger("finetuning.evaluation.evaluate")


def _construct_mewsli_path(root_dir: str, finetuning_round: int, lang: str) -> str:
    return f"{root_dir}/mewsli_embs_{lang}_{finetuning_round}"


def _construct_damuel_path(root_dir: str, finetuning_round: int) -> str:
    next_finetuning_round = finetuning_round + 1
    return f"{root_dir}/damuel_for_index_{next_finetuning_round}"


def evaluate(
    root_dir: str,
    finetuning_round: int,
    langs: list[str] = ["ar", "de", "en", "es", "ja", "fa", "sr", "ta", "tr"],
):
    damuel_path = _construct_damuel_path(root_dir, finetuning_round)

    damuel_embs, damuel_qids = load_embs_and_qids_with_normalization(damuel_path)
    searcher = BruteForceSearcher(damuel_embs, damuel_qids)

    for lang in langs:
        mewsli_path = _construct_mewsli_path(root_dir, finetuning_round, lang)
        _logger.info(f"Calculating recall for {lang}")
        find_recall_with_searcher(searcher, mewsli_path, _RECALLS)
