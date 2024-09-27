from pathlib import Path
import logging

_logger = logging.getLogger(__name__)

import gin

from utils.embeddings import embs_from_tokens_and_model_name

from .at_embeddings import embs_from_tokens_and_model_name_at
from .find_recall import find_recall


def create_embeddings(damuel_tokens, mewsli_tokens, workdir, language):
    """
    Create embeddings for the given language.
    """
    if not (workdir / language / "damuel" / "embs_tokens.npz").exists():
        embs_from_tokens_and_model_name_at(
            source=damuel_tokens / language,
            dest=workdir / language / "damuel",
        )
    if not (workdir / language / "mewsli" / "embs_qids.npz").exists():
        embs_from_tokens_and_model_name(
            source=mewsli_tokens / language,
            dest=workdir / language / "mewsli",
        )


def evaluate_olpeat(damuel_tokens, workdir, language, recalls):
    """
    Evaluate OLPEAT for the given language.
    """
    for recall in recalls:
        find_recall(
            workdir / language / "damuel" / "embs_tokens.npz",
            damuel_tokens / language,
            workdir / language / "mewsli" / "embs_qids.npz",
            recall,
        )


@gin.configurable
def olpeat(damuel_tokens, mewsli_tokens, workdir, recalls, languages):
    """
    High-level OLPEAT function that does for all languages from the gin config the following:
        - Creates embeddings
        - Evaluates OLPEAT

    All the work is done in the workdir where appropriate subdirectories are created.
    The method does not clean after itself, so the output directories are not deleted.
    """
    damuel_tokens = Path(damuel_tokens)
    mewsli_tokens = Path(mewsli_tokens)
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    for language in languages:
        # mkdir lang directory
        _logger.info(f"Solving for language {language}")
        (workdir / language).mkdir(parents=True, exist_ok=True)
        (workdir / language / "damuel").mkdir(parents=True, exist_ok=True)
        (workdir / language / "mewsli").mkdir(parents=True, exist_ok=True)

        # Create embeddings
        create_embeddings(damuel_tokens, mewsli_tokens, workdir, language)

        # Evaluate OLPEAT
        evaluate_olpeat(damuel_tokens, workdir, language, recalls)
