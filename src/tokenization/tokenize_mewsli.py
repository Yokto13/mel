from pathlib import Path

from tokenization.generate_tokens import (
    tokens_for_finetuning_mewsli,
    tokens_for_at_mewsli,
)


def process(p: Path, out, workers, context_size, model_name, ignore_context):
    out_lang_path = out / p.name
    out_lang_path.mkdir(parents=True, exist_ok=True)
    if ignore_context:
        tokens_for_at_mewsli(
            model_name, p / "mentions.tsv", context_size, out_lang_path, workers
        )
    else:
        tokens_for_finetuning_mewsli(
            model_name, p / "mentions.tsv", context_size, out_lang_path, workers
        )


def tokens_for_all_mewsli(
    mewsli, out, workers, context_size, model_name, ignore_context
):
    mewsli = Path(mewsli)
    out = Path(out)

    for p in mewsli.iterdir():
        if not p.is_dir() or "wikidata" in p.name:
            continue
        process(p, out, workers, context_size, model_name, ignore_context)
