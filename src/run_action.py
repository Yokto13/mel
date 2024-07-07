from functools import partial

from fire import Fire
import wandb

from baselines.alias_table.all_languages import all_languages
from baselines.alias_table.one_language_lemma import alias_table_with_lemmas
from baselines.alias_table.from_tokens import one_language
from baselines.alias_table.string_similarity import string_similarity

from data_processors.tokens.duplicates_filter_script import run_duplicates_filter_script

from finetunings.embs_generating.build_together_embs import generate_embs
from finetunings.token_index.save_token_index import build_and_save_token_index
from finetunings.generate_epochs.generate import generate
from finetunings.finetune_model.train import train
from finetunings.evaluation.evaluate import evaluate, run_recall_calculation
from finetunings.file_processing.gathers import move_tokens, rename, remove_duplicates

print("Importing problematic part")
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
tokens_for_all_mewsli_finetuning = partial(tokens_for_all_mewsli, ignore_context=False)
tokens_for_all_damuel_finetuning = partial(tokens_for_all_damuel, ignore_context=False)

from utils.extractors.orchestrator import damuel_description_tokens

print("Imports finished")


def choose_action(action):
    match action:
        case "embs":
            return generate_embs
        case "token_index":
            return build_and_save_token_index
        case "generate":
            return generate
        case "train":
            return train
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
        case "tokens_for_all_mewsli_finetuning":
            return tokens_for_all_mewsli_finetuning
        case "tokens_for_all_damuel_finetuning":
            return tokens_for_all_damuel_finetuning
        case _:
            raise ValueError(f"Unknown action: {action}")


def main(*args):
    print("hello")
    wandb.init(project=f"test-{args[0]}", config={"action": args[0], "args": args[1:]})
    action_descriptor = args[0]
    action = choose_action(action_descriptor)
    print(f"Running {action_descriptor} with args {args[1:]}")
    action(*args[1:])


if __name__ == "__main__":
    Fire(main)
