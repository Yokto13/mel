import logging

from finetunings.evaluation.find_recall import find_recall

_RECALLS = [1, 10, 100]

_logger = logging.getLogger("finetuning.evaluation.evaluate")


def _construct_mewsli_path(root_dir: str, finetuning_round: int, lang: str) -> str:
    return f"{root_dir}/mewsli_embs_{lang}_{finetuning_round}"


def _construct_damuel_path(root_dir: str, finetuning_round: int) -> str:
    next_finetuning_round = finetuning_round + 1
    return f"{root_dir}/damuel_for_index_{next_finetuning_round}"


def run_recall_calculation(damuel_dir, mewsli_dir, recall=None):
    recalls = _RECALLS if recall is None else [recall]
    find_recall(damuel_dir, mewsli_dir, recalls)


def evaluate(
    root_dir: str,
    finetuning_round: int,
    langs: list[str] = ["ar", "de", "en", "es", "ja", "fa", "sr", "ta", "tr"],
):
    damuel_path = _construct_damuel_path(root_dir, finetuning_round)

    for lang in langs:
        mewsli_path = _construct_mewsli_path(root_dir, finetuning_round, lang)
        _logger.info(f"Calculating recall for {lang}")
        run_recall_calculation(damuel_path, mewsli_path)
