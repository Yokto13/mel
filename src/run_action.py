import logging
from functools import partial

logging.basicConfig(level=logging.INFO)

import wandb

from baselines.alias_table.all_languages import all_languages
from baselines.alias_table.from_tokens import one_language
from baselines.alias_table.one_language_lemma import alias_table_with_lemmas
from baselines.alias_table.string_similarity import string_similarity
from baselines.olpeat.at_embeddings import embs_from_tokens_and_model_name_at
from baselines.olpeat.find_recall import find_recall as find_recall_olpeat
from finetunings.evaluation.evaluate import evaluate, run_recall_calculation
from finetunings.file_processing.gathers import move_tokens, remove_duplicates, rename
from finetunings.finetune_model.train import train
from finetunings.finetune_model.train_ddp import train_ddp
from finetunings.generate_epochs.embed_links_for_generation import (
    embed_links_for_generation,
)

from finetunings.generate_epochs.generate import generate

# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

from fire import Fire
from multilingual_dataset.combine_embs import combine_embs_by_qid

from multilingual_dataset.creator import create_multilingual_dataset, run_kb_creator

from utils.arg_names import get_args_names

from utils.embeddings import (
    embs_from_tokens_and_model_name,
    embs_from_tokens_model_name_and_state_dict,
)
from utils.validate_tokens import validate_tokens

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
        case "rename":
            return rename
        case "remove_duplicates":
            return remove_duplicates
        case "filter_duplicates_script":
            return run_duplicates_filter_script
        case "embs_from_tokens_and_model_name":
            return embs_from_tokens_and_model_name
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
        case "run_kb_creator":
            return run_kb_creator
        case "combine_embs_by_qid":
            return combine_embs_by_qid
        case "run_mewsli_mention":
            return run_mewsli_mention
        case "run_mewsli_context":
            return run_mewsli_context
        case "validate_tokens":
            return validate_tokens
        case "run_damuel_description_mention":
            return run_damuel_description_mention
        case "run_damuel_description_context":
            return run_damuel_description_context
        case "run_damuel_link_context":
            return run_damuel_link_context
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
