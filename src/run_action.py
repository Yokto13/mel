from functools import partial
import logging

logging.basicConfig(level=logging.INFO)

# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

from fire import Fire
import wandb

from baselines.alias_table.all_languages import all_languages
from baselines.alias_table.one_language_lemma import alias_table_with_lemmas
from baselines.alias_table.from_tokens import one_language
from baselines.alias_table.string_similarity import string_similarity
from baselines.olpeat.at_embeddings import embs_from_tokens_and_model_name_at
from baselines.olpeat.meludr_olpeat import meludr_olpeat
from baselines.olpeat.meludr_evaluate import meludr_run_recall_calculation
from baselines.olpeat.find_recall import find_recall as find_recall_olpeat

from utils.embeddings import (
    embs_from_tokens_and_model_name,
    embs_from_tokens_model_name_and_state_dict,
)

from data_processors.tokens.duplicates_filter_script import run_duplicates_filter_script

from finetunings.generate_epochs.generate import generate
from finetunings.generate_epochs.embed_links_for_generation import (
    embed_links_for_generation,
)
from finetunings.finetune_model.train import train
from finetunings.finetune_model.train_ddp import train_ddp
from finetunings.evaluation.evaluate import evaluate, run_recall_calculation
from finetunings.file_processing.gathers import move_tokens, rename, remove_duplicates

from multilingual_dataset.creator import create_multilingual_dataset

from tokenization.generate_tokens import (
    tokens_for_finetuning_mewsli,
    tokens_for_finetuning_damuel_descriptions,
    tokens_for_finetuning_damuel_links,
    tokens_for_at_descriptions,
    tokens_for_at_links,
    tokens_for_at_mewsli,
)

from tokenization.tokenize_mewsli import tokens_for_all_mewsli
from tokenization.tokenize_damuel import tokens_for_all_damuel


tokens_for_all_mewsli_at = partial(tokens_for_all_mewsli, ignore_context=True)
tokens_for_all_damuel_at = partial(tokens_for_all_damuel, ignore_context=True)
tokens_for_all_damuel_at_pages = partial(
    tokens_for_all_damuel, ignore_context=True, only_pages=True
)
tokens_for_all_mewsli_finetuning = partial(tokens_for_all_mewsli, ignore_context=False)
tokens_for_all_damuel_finetuning = partial(tokens_for_all_damuel, ignore_context=False)
tokens_for_all_damuel_finetuning_pages = partial(
    tokens_for_all_damuel, ignore_context=False, only_pages=True
)

from utils.extractors.orchestrator import damuel_description_tokens
from utils.arg_names import get_args_names

print("Imports finished")


def choose_action(action):
    match action:
        case "generate":
            return generate
        case "train":
            return train
        case "train_ddp":
            return train_ddp
        case "evaluate":
            return evaluate
        case "copy":
            return move_tokens
        case "at_lemmas":
            return alias_table_with_lemmas
        case "at_one":
            return one_language
        case "at_all":
            return all_languages
        case "string_similarity":
            return string_similarity
        case "recalls":
            return run_recall_calculation
        case "meludr_recalls":
            return meludr_run_recall_calculation
        case "rename":
            return rename
        case "remove_duplicates":
            return remove_duplicates
        case "filter_duplicates_script":
            return run_duplicates_filter_script
        case "tokens_mewsli":
            return tokens_for_finetuning_mewsli
        case "tokens_descriptions":
            return tokens_for_finetuning_damuel_descriptions
        case "tokens_descriptions_new":
            return damuel_description_tokens
        case "tokens_links":
            return tokens_for_finetuning_damuel_links
        case "tokens_descriptions_at":
            return tokens_for_at_descriptions
        case "tokens_links_at":
            return tokens_for_at_links
        case "tokens_mewsli_at":
            return tokens_for_at_mewsli
        case "tokens_for_all_mewsli_at":
            return tokens_for_all_mewsli_at
        case "tokens_for_all_damuel_at":
            return tokens_for_all_damuel_at
        case "tokens_for_all_damuel_at_pages":
            return tokens_for_all_damuel_at_pages
        case "tokens_for_all_mewsli_finetuning":
            return tokens_for_all_mewsli_finetuning
        case "tokens_for_all_damuel_finetuning":
            return tokens_for_all_damuel_finetuning
        case "tokens_for_all_damuel_finetuning_pages":
            return tokens_for_all_damuel_finetuning_pages
        case "embs_from_tokens_and_model_name":
            return embs_from_tokens_and_model_name
        case "meludr_olpeat":
            return meludr_olpeat
        case "embs_from_tokens_and_model_name_at":
            return embs_from_tokens_and_model_name_at
        case "find_recall_olpeat":
            return find_recall_olpeat
        case "embs_from_tokens_model_name_and_state_dict":
            return embs_from_tokens_model_name_and_state_dict
        case "embed_links_for_generation":
            return embed_links_for_generation
        case "create_multilingual_dataset":
            return create_multilingual_dataset
        case _:
            raise ValueError(f"Unknown action: {action}")


def main(*args):
    print("hello")
    action_descriptor = args[0]
    action = choose_action(action_descriptor)
    arg_names = get_args_names(action)
    print(arg_names)
    wandb.init(
        project=f"EL-{args[0]}",
        config={"action": args[0]}
        | {arg_name: arg_value for arg_name, arg_value in zip(arg_names, args[1:])},
    )
    print(f"Running {action_descriptor} with args {args[1:]}")
    action(*args[1:])


if __name__ == "__main__":
    Fire(main)
