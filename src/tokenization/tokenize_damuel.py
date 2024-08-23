from enum import Enum, auto
from pathlib import Path
from tqdm import tqdm

from tokenization.generate_tokens import (
    tokens_for_finetuning_damuel_descriptions,
    tokens_for_finetuning_damuel_descriptions_pages,
    tokens_for_finetuning_damuel_links,
    tokens_for_at_descriptions,
    tokens_for_at_descriptions_pages,
    tokens_for_at_links,
)


def _get_lang_from_name(name):
    return name.split("_")[-1]


class _DamuelGenerationType(Enum):
    IGNORE_CONTEXT = auto()
    IGNORE_CONTEXT_PAGES = auto()
    FINETUNING = auto()
    FINETUNING_PAGES = auto()


def _process(
    p: Path,
    out,
    workers,
    context_size,
    model_name,
    generation_type: _DamuelGenerationType,
):
    # print("Procesing", p)
    out_lang_path = out / _get_lang_from_name(p.name)
    out_lang_path.mkdir(parents=True, exist_ok=True)
    d = out_lang_path / "descs"
    d.mkdir(parents=True, exist_ok=True)
    l = out_lang_path / "links"
    l.mkdir(parents=True, exist_ok=True)
    dp = out_lang_path / "descs_pages"
    dp.mkdir(parents=True, exist_ok=True)
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
            # TODO: there should be more freedom over choosing the generation types.
            # Currently the Iterators sometimes disregard SRP which makes any custom tokenization very hard.
            tokens_for_at_descriptions_pages(model_name, p, context_size, d, workers)
        case _DamuelGenerationType.FINETUNING_PAGES:
            print("Finetuning pages!!")
            tokens_for_finetuning_damuel_descriptions_pages(
                model_name, p, context_size, dp, workers
            )


def _choose_generation_type(ignore_context, only_pages):
    generation_type = (
        _DamuelGenerationType.IGNORE_CONTEXT
        if ignore_context
        else _DamuelGenerationType.FINETUNING
    )
    if generation_type == _DamuelGenerationType.FINETUNING and only_pages:
        return _DamuelGenerationType.FINETUNING_PAGES
    if generation_type == _DamuelGenerationType.IGNORE_CONTEXT and only_pages:
        generation_type = _DamuelGenerationType.IGNORE_CONTEXT_PAGES
    return generation_type


def tokens_for_all_damuel(
    damuel, out, workers, context_size, model_name, ignore_context, only_pages=False
):
    """Tokenizes the whole DaMuEL. Parameters constrain the tokenization.

    Args:
        damuel (str): path to directory containing directory per each DaMuEL language.
        out (str): output path.
        workers (int): number of workers to use. The computation utilizes subprocess, should always be <= 500.
        context_size (int): Then number of tokens that the result will have.
        model_name (str): Path to the model.
        ignore_context (bool): If True, tokenizes only the label/title.
        only_pages (bool, optional): When a language specic entry lacks a page it is skipped.
            This currently generates only description tokens and requires ignore_context=True.
            Defaults to False.
    """
    damuel = Path(damuel)
    out = Path(out)

    generation_type = _choose_generation_type(ignore_context, only_pages)
    print(generation_type)

    for p in tqdm(list(damuel.iterdir())):
        if not p.is_dir() or "wikidata" in p.name:
            continue
        try:
            _process(p, out, workers, context_size, model_name, generation_type)
        except FileExistsError as e:
            print(p, "exists... Skipping!")
            print(e)
