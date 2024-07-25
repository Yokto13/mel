from enum import Enum, auto
from pathlib import Path
from tqdm import tqdm

from tokenization.generate_tokens import (
    tokens_for_finetuning_damuel_descriptions,
    tokens_for_finetuning_damuel_links,
    tokens_for_at_descriptions,
    tokens_for_at_descriptions_pages,
    tokens_for_at_links,
)


def _get_lang_from_name(name):
    return name.split('_')[-1]


class _DamuelGenerationType(Enum):
    IGNORE_CONTEXT=auto()
    IGNORE_CONTEXT_PAGES=auto()
    FINETUNING=auto()


def process(p: Path, out, workers, context_size, model_name, generation_type: _DamuelGenerationType):
    # print("Procesing", p)
    out_lang_path = out / _get_lang_from_name(p.name)
    out_lang_path.mkdir(parents=True, exist_ok=False)
    d = out_lang_path / "descs"
    d.mkdir(parents=True, exist_ok=True)
    l = out_lang_path / "links"
    l.mkdir(parents=True, exist_ok=True)
    match generation_type:
        case _DamuelGenerationType.IGNORE_CONTEXT:
            tokens_for_at_descriptions(model_name, p, context_size, d, workers)
            tokens_for_at_links(model_name, p, context_size, l, workers)
        case _DamuelGenerationType.FINETUNING:
            tokens_for_finetuning_damuel_descriptions(
                model_name, p, context_size, d, workers
            )
            tokens_for_finetuning_damuel_links(model_name, p, context_size, l, workers)
        case _DamuelGenerationType.IGNORE_CONTEXT_PAGES:
            tokens_for_at_descriptions_pages(model_name, p, context_size, d, workers)


def _choose_generation_type(ignore_context, only_pages):
    generation_type = _DamuelGenerationType.IGNORE_CONTEXT if ignore_context else _DamuelGenerationType.FINETUNING
    if generation_type == _DamuelGenerationType.IGNORE_CONTEXT and only_pages:
        generation_type = _DamuelGenerationType.IGNORE_CONTEXT_PAGES
    return generation_type


def tokens_for_all_damuel(
    damuel, out, workers, context_size, model_name, ignore_context, only_pages=False
):
    damuel = Path(damuel)
    out = Path(out)

    generation_type = _choose_generation_type(ignore_context, only_pages)
    print(generation_type)

    for p in tqdm(list(damuel.iterdir())):
        if not p.is_dir() or "wikidata" in p.name:
            continue
        try:
            process(p, out, workers, context_size, model_name, generation_type)
        except FileExistsError:
            print(p, "exists... Skipping!")
