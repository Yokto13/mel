import logging

logging.basicConfig(level=logging.INFO)

from fire import Fire
import gin
import wandb

from baselines.string_alias_tables.all_languages import all_languages
from baselines.olpeat.at_embeddings import embs_from_tokens_and_model_name_at
from baselines.olpeat.find_recall import find_recall as find_recall_olpeat
from baselines.olpeat.olpeat import olpeat
from finetunings.evaluation.evaluate import evaluate, run_recall_calculation
from finetunings.evaluation.find_recall import find_candidates
from finetunings.file_processing.gathers import move_tokens, remove_duplicates, rename
from finetunings.finetune_model.train import train
from finetunings.finetune_model.train_ddp import train_ddp
from finetunings.generate_epochs.embed_links_for_generation import (
    embed_links_for_generation,
)
from finetunings.generate_epochs.generate import generate
from multilingual_dataset.combine_embs import combine_embs_by_qid
from multilingual_dataset.creator import create_multilingual_dataset, run_kb_creator
from tokenization.runner import (
    run_damuel_description_context,
    run_damuel_description_mention,
    run_damuel_link_context,
    run_damuel_link_mention,
    run_damuel_mention,
    run_mewsli_context,
    run_mewsli_mention,
)
from utils.arg_names import get_args_names
from utils.embeddings import (
    embs_from_tokens_and_model_name,
    embs_from_tokens_model_name_and_state_dict,
)
from utils.validate_tokens import validate_tokens

print("Imports finished")


def choose_action(action):
    match action:
        case "standard_alias_table":
            return all_languages
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
        case "recalls":
            return run_recall_calculation
        case "rename":
            return rename
        case "remove_duplicates":
            return remove_duplicates
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
        case "run_damuel_link_mention":
            return run_damuel_link_mention
        case "run_damuel_mention":
            return run_damuel_mention
        case "olpeat":
            return olpeat
        case "find_candidates":
            return find_candidates
        case _:
            raise ValueError(f"Unknown action: {action}")


def main(*args, **kwargs):
    gin.parse_config_file("../configs/general.gin")
    action_descriptor = args[-1]
    action = choose_action(action_descriptor)
    arg_names = get_args_names(action)
    print(arg_names)
    wandb.init(
        project=f"EL-{args[-1]}",
        config={"action": args[-1]}
        | {arg_name: arg_value for arg_name, arg_value in zip(arg_names, args[:-1])},
        notes=str(args[:-1]),
    )
    print(f"Running {action_descriptor} with args {args[:-1]} and kwargs {kwargs}")
    for arg in args[:-1]:
        print("parsing arg", arg)
        gin.parse_config_file(arg)
    action(**kwargs)


if __name__ == "__main__":
    Fire(main)
