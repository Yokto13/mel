import os
from pathlib import Path

from finetunings.evaluation.evaluate import evaluate
from finetunings.file_processing.gathers import move_tokens
from utils.argument_wrappers import ensure_datatypes


# Relict of old naming, the subsequent methods expect 'mentions' as file names
def _rename_entity_names_to_mentions(dir: Path):
    for file in os.listdir(dir):
        if file.startswith("entity_names"):
            os.rename(
                Path(dir, file), Path(dir, file.replace("entity_names", "mentions"))
            )


def make_dirs(dirs):
    for dir in dirs:
        os.makedirs(dir, exist_ok=True)


@ensure_datatypes([Path, Path, Path, Path, str], {})
def one_language(
    raw_mewsli: Path, raw_links: Path, raw_desc: Path, workdir: Path, model_path: str
):
    mewsli_tokens = Path(workdir, "mewsli_tokens")
    damuel_tokens = Path(workdir, "damuel_tokens")
    mewsli_dir = Path(workdir, "mewsli_embs")
    damuel_dir = Path(workdir, "damuel_embs")

    make_dirs([mewsli_tokens, damuel_tokens, mewsli_dir, damuel_dir])

    move_tokens(raw_mewsli, mewsli_tokens)
    _rename_entity_names_to_mentions(mewsli_tokens)

    move_tokens(raw_desc, damuel_tokens)
    move_tokens(raw_links, damuel_tokens)
    _rename_entity_names_to_mentions(damuel_tokens)

    evaluate(damuel_tokens, mewsli_tokens, model_path, damuel_dir, mewsli_dir)
