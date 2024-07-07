# from multiprocessing import set_start_method
# set_start_method("spawn")
from functools import partial
import fire
from math import inf
import multiprocessing
from enum import Enum, auto

from transformers import BertTokenizerFast
import numpy as np

from data_processors.tokens.damuel.descriptions.both import (
    DamuelDescriptionsTokensIteratorBoth,
)
from data_processors.tokens.damuel.links.both import DamuelLinksTokensIteratorBoth
from data_processors.tokens.damuel.descriptions.for_finetuning import (
    DamuelDescriptionsTokensIteratorFinetuning,
)
from data_processors.tokens.damuel.links.for_finetuning import (
    DamuelLinksTokensIteratorFinetuning,
)

from data_processors.tokens.mewsli.tokens_iterator_both import MewsliTokensIteratorBoth
from src.data_processors.tokens.mewsli.for_finetuning import (
    MewsliTokensIteratorFinetuning,
)

per_save = 10**5


class GenerationType(Enum):
    FINETUNING_LINKS = auto()
    FINETUNING_DESCRIPTIONS = auto()
    FINETUNING_MEWSLI = auto()
    OLD_MEWSLI = auto()
    OLD_LINKS = auto()
    OLD_DESCRIPTIONS = auto()
    MENTIONS_LINKS = auto()
    MENTIONS_DESCRIPTIONS = auto()
    MENTIONS_MEWSLI = auto()


def save_token_qid_pairs(pairs, output_path):
    print("Saving", len(pairs), "items")
    tokens = np.empty((len(pairs), len(pairs[0][0])), dtype=np.uint16)
    qids = np.empty(len(pairs), dtype=np.uint32)
    for i in range(len(pairs)):
        # print(pairs[i][0])
        tokens[i] = pairs[i][0]
        qids[i] = pairs[i][1]
    np.savez_compressed(output_path, tokens=tokens, qids=qids)


def entity_names_save(entity_names, mentions, output_dir):
    hv = abs(hash(abs(hash(str(mentions[0]))) + abs(hash(str(mentions[-1])))))

    print(f"Saving to file {hv}")

    save_token_qid_pairs(entity_names, str(output_dir) + f"/entity_names_{hv}.npz")
    save_token_qid_pairs(mentions, str(output_dir) + f"/mentions_{hv}.npz")


def mentions_save(mentions, output_dir, name="mentions"):
    hv = abs(hash(abs(hash(str(mentions[0]))) + abs(hash(str(mentions[-1])))))

    print(f"Saving to file {hv}")

    print(f"" + str(output_dir) + f"/{name}_{hv}.npz")
    save_token_qid_pairs(mentions, str(output_dir) + f"/{name}_{hv}.npz")


def get_iterator_class(generation_type):
    match generation_type:
        case GenerationType.MENTIONS_LINKS | GenerationType.OLD_LINKS:
            return DamuelLinksTokensIteratorBoth
        case GenerationType.OLD_DESCRIPTIONS | GenerationType.MENTIONS_DESCRIPTIONS:
            return DamuelDescriptionsTokensIteratorBoth
        case GenerationType.FINETUNING_LINKS:
            return DamuelLinksTokensIteratorFinetuning
        case GenerationType.FINETUNING_DESCRIPTIONS:
            return DamuelDescriptionsTokensIteratorFinetuning
        case GenerationType.FINETUNING_MEWSLI:
            return MewsliTokensIteratorFinetuning
        case GenerationType.OLD_MEWSLI | GenerationType.MENTIONS_MEWSLI:
            return MewsliTokensIteratorBoth


def is_part_good_for_iterator(part, workers, r):
    if "." in part:
        part = part.split(".")[0]
    return int(part.split("-")[1]) % workers == r


def get_iterators(args, kwargs, iterator_class, workers):
    if (
        iterator_class == MewsliTokensIteratorBoth
        or iterator_class == MewsliTokensIteratorFinetuning
    ):
        if "only_wiki" in kwargs:
            del kwargs["only_wiki"]
        yield iterator_class(*args, **kwargs)
    else:
        for i in range(workers):
            print(i)
            part_f = partial(is_part_good_for_iterator, workers=workers, r=i)
            if (
                iterator_class == DamuelDescriptionsTokensIteratorBoth
                or iterator_class == DamuelDescriptionsTokensIteratorFinetuning
            ):
                if "only_wiki" in kwargs:
                    del kwargs["only_wiki"]
                yield iterator_class(*args, **kwargs, filename_is_ok=part_f)
            else:
                yield iterator_class(*args, **kwargs, filename_is_ok=part_f)


def solve(iterator, output_dir):
    entity_names = []
    mentions = []

    for entity_name, context in iterator:
        entity_names.append(entity_name)
        mentions.append(context)

        if len(entity_names) == per_save:
            entity_names_save(entity_names, mentions, output_dir)
            entity_names = []
            mentions = []

    entity_names_save(entity_names, mentions, output_dir)


def solve_only_contexts(iterator, output_dir):
    mentions = []

    for context in iterator:
        mentions.append(context)

        if len(mentions) == per_save:
            mentions_save(mentions, output_dir)
            mentions = []

    mentions_save(mentions, output_dir)


def solve_only_names(iterator, output_dir):
    entity_names = []
    for entity_name, context in iterator:
        entity_names.append(entity_name)

        if len(entity_names) == per_save:
            mentions_save(entity_names, output_dir, name="entity_names")
            entity_names = []

    mentions_save(entity_names, output_dir, name="entity_names")


def main(
    model_name,
    data_path,
    context_size,
    generation_type: GenerationType,
    output_dir,
    workers=1,
):
    tokenizer = BertTokenizerFast.from_pretrained(model_name)
    tokenizer.add_tokens("[M]")

    iterator_class = get_iterator_class(generation_type)

    iterators = list(
        get_iterators(
            (data_path, tokenizer),
            {
                "expected_size": context_size,
                "only_wiki": True,
            },
            iterator_class,
            workers,
        )
    )

    match generation_type:
        case (
            GenerationType.FINETUNING_DESCRIPTIONS
            | GenerationType.FINETUNING_LINKS
            | GenerationType.FINETUNING_MEWSLI
        ):
            solve_f = solve_only_contexts
        case (
            GenerationType.MENTIONS_DESCRIPTIONS
            | GenerationType.MENTIONS_LINKS
            | GenerationType.MENTIONS_MEWSLI
        ):
            solve_f = solve_only_names
        case _:
            solve_f = solve

    solve_with_output = partial(solve_f, output_dir=output_dir)

    print(f"Running with {workers} workers")

    with multiprocessing.Pool(workers) as p:
        p.map(solve_with_output, iterators)


def tokens_for_finetuning_mewsli(
    model_name, data_path, context_size, output_dir, workers
):
    run_type = GenerationType.FINETUNING_MEWSLI
    main(model_name, data_path, context_size, run_type, output_dir, workers)


def tokens_for_finetuning_damuel_descriptions(
    model_name, data_path, context_size, output_dir, workers
):
    run_type = GenerationType.FINETUNING_DESCRIPTIONS
    main(model_name, data_path, context_size, run_type, output_dir, workers)


def tokens_for_finetuning_damuel_links(
    model_name, data_path, context_size, output_dir, workers
):
    run_type = GenerationType.FINETUNING_LINKS
    main(model_name, data_path, context_size, run_type, output_dir, workers)


def tokens_for_at_links(model_name, data_path, context_size, output_dir, workers):
    run_type = GenerationType.MENTIONS_LINKS
    main(model_name, data_path, context_size, run_type, output_dir, workers)


def tokens_for_at_mewsli(model_name, data_path, context_size, output_dir, workers):
    run_type = GenerationType.MENTIONS_MEWSLI
    main(model_name, data_path, context_size, run_type, output_dir, workers)


def tokens_for_at_descriptions(
    model_name, data_path, context_size, output_dir, workers
):
    run_type = GenerationType.MENTIONS_DESCRIPTIONS
    main(model_name, data_path, context_size, run_type, output_dir, workers)


if __name__ == "__main__":
    fire.Fire(main)
