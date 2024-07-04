# from multiprocessing import set_start_method
# set_start_method("spawn")
from functools import partial
import lzma
import sys
import fire
import pickle
from math import inf
import multiprocessing

from transformers import BertTokenizerFast

sys.path.append("/home/farhand/bc/src")

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

per_save = 10**4


def entity_names_save(entity_names, mentions, output_dir):
    hv = abs(hash(abs(hash(mentions[0])) + abs(hash(mentions[-1]))))

    print(f"Saving to file {hv}")

    with lzma.open(f"" + output_dir + f"/entity_names_{hv}.xz", "wb") as f:
        pickle.dump(entity_names, f)
    with lzma.open(f"" + output_dir + f"/mentions_{hv}.xz", "wb") as f:
        pickle.dump(mentions, f)


def mentions_save(mentions, output_dir, name="mentions"):
    hv = abs(hash(abs(hash(mentions[0])) + abs(hash(mentions[-1]))))

    print(f"Saving to file {hv}")

    print(f"" + output_dir + f"/{name}_{hv}.xz")
    with lzma.open(f"" + output_dir + f"/{name}_{hv}.xz", "wb") as f:
        pickle.dump(mentions, f)


def get_iterator_class(type):
    if type == "links" or type == "links_names":
        return DamuelLinksTokensIteratorBoth
    elif type == "descs" or type == "descs_names":
        return DamuelDescriptionsTokensIteratorBoth
    if type == "links_together":
        return DamuelLinksTokensIteratorFinetuning
    elif type == "descs_together":
        return DamuelDescriptionsTokensIteratorFinetuning
    elif type == "mewsli_together":
        return MewsliTokensIteratorFinetuning
    elif type == "mewsli" or type == "mewsli_names":
        return MewsliTokensIteratorBoth


def is_part_good_for_iterator(part, workers, r):
    if "." in part:
        part = part.split(".")[0]
    return int(part.split("-")[1]) % workers == r


def get_iterators(args, kwargs, iterator_class, workers):
    if (
        iterator_class == MewsliTokensIteratorBoth
        or iterator_class == MewsliTokensIteratorFinetuning
        or iterator_class == DamuelDescriptionsTokensIteratorBoth
        or iterator_class == DamuelDescriptionsTokensIteratorFinetuning
    ):
        if "only_wiki" in kwargs:
            del kwargs["only_wiki"]
        yield iterator_class(*args, **kwargs)
    else:
        for i in range(workers):
            print(i)
            part_f = partial(is_part_good_for_iterator, workers=workers, r=i)
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


def main(model_name, data_path, context_size, type, output_dir, workers=1):
    tokenizer = BertTokenizerFast.from_pretrained(model_name)

    iterator_class = get_iterator_class(type)

    iterators = list(
        get_iterators(
            (data_path, tokenizer),
            {
                "expected_size": context_size,
                "only_wiki": True,
                "treat_qids_as_ints": True,
            },
            iterator_class,
            workers,
        )
    )

    if "together" in type:
        solve_f = solve_only_contexts
    elif "names" in type:
        solve_f = solve_only_names
    else:
        solve_f = solve

    solve_with_output = partial(solve_f, output_dir=output_dir)

    print(f"Running with {workers} workers")

    with multiprocessing.Pool(workers) as p:
        p.map(solve_with_output, iterators)


def tokens_for_finetuning_mewsli(
    model_name, data_path, context_size, output_dir, workers
):
    run_type = "mewsli_together"
    main(model_name, data_path, context_size, run_type, output_dir, workers)


def tokens_for_finetuning_damuel_descriptions(
    model_name, data_path, context_size, output_dir, workers
):
    run_type = "descs_together"
    main(model_name, data_path, context_size, run_type, output_dir, workers)


def tokens_for_finetuning_damuel_links(
    model_name, data_path, context_size, output_dir, workers
):
    run_type = "links_together"
    main(model_name, data_path, context_size, run_type, output_dir, workers)


if __name__ == "__main__":
    fire.Fire(main)
