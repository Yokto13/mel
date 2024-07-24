from pathlib import Path
from tqdm import tqdm

from tokenization.generate_tokens import (
    tokens_for_finetuning_damuel_descriptions,
    tokens_for_finetuning_damuel_links,
    tokens_for_at_descriptions,
    tokens_for_at_links,
)


def process(p: Path, out, workers, context_size, model_name, ignore_context):
    # print("Procesing", p)
    out_lang_path = out / p.name
    out_lang_path.mkdir(parents=True, exist_ok=False)
    d = out_lang_path / "descs"
    d.mkdir(parents=True, exist_ok=True)
    l = out_lang_path / "links"
    l.mkdir(parents=True, exist_ok=True)
    if ignore_context:
        tokens_for_at_descriptions(model_name, p, context_size, d, workers)
        tokens_for_at_links(model_name, p, context_size, l, workers)
    else:
        tokens_for_finetuning_damuel_descriptions(
            model_name, p, context_size, d, workers
        )
        tokens_for_finetuning_damuel_links(model_name, p, context_size, l, workers)


def tokens_for_all_damuel(
    damuel, out, workers, context_size, model_name, ignore_context
):
    damuel = Path(damuel)
    out = Path(out)

    for p in tqdm(list(damuel.iterdir())):
        if not p.is_dir() or "wikidata" in p.name:
            continue
        try:
            process(p, out, workers, context_size, model_name, ignore_context)
        except FileExistsError:
            print(p, "exists... Skipping!")
