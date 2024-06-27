from collections import deque
from dataclasses import dataclass
from copy import deepcopy
import lzma
from pathlib import Path
import pickle
import sys
import blosc

sys.stdout.reconfigure(line_buffering=True, write_through=True)

from fire import Fire
import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm

from transformers import BertModel, BertTokenizerFast

import wandb

from models.finetuning.wrapper import FinetuningWrapper
from models.finetuning.wrapper_with_multiplier import FinetuningWrapperWithMultiplier
from models.finetuning.wrapper_gillick_like import FinetuningWrapperGillick
from utils.argument_wrappers import ensure_datatypes
from utils.running_averages import RunningAverages

# Settings ===========================================

_RUNNING_AVERAGE_SMALL = 100
_RUNNING_AVERAGE_BIG = 1000


if torch.cuda.is_available():
    print("CUDA is available!")
    device = torch.device("cuda")
else:
    print("CUDA is not available.")
    device = torch.device("cpu")

SEED = 0
torch.manual_seed(SEED)


@dataclass
class SaveInformation:
    type: str
    output_path: Path
    is_final: bool
    epoch: int = None
    recall: int = None


def load_epoch_xz(path, epoch):
    with lzma.open(path / f"epoch_{epoch}.pkl.xz", "rb") as f:
        return pickle.load(f)


def load_epoch_blosc(path, epoch):
    with open(path / f"epoch_{epoch}.dat", "rb") as f:
        compressed = f.read()
    decompressed = blosc.decompress(compressed)
    return pickle.loads(decompressed)


def batch_recall(outputs, target, k=1):
    _, top_indices = outputs.topk(k, dim=-1)
    top_values = target.gather(-1, top_indices)
    recall_per_row = top_values.any(dim=-1).float()
    return recall_per_row.mean()


def gillick_loss(similarities, targets, model: FinetuningWrapperGillick):
    logits_for_cross_entropy = similarities * model.softmax_multiplier
    cross_entropy_component = nn.CrossEntropyLoss()(logits_for_cross_entropy, targets)
    logits_for_binary = model.sigmoid_multiplier * similarities + model.sigmoid_offset
    binary_component = nn.BCEWithLogitsLoss()(logits_for_binary, targets)
    return cross_entropy_component + binary_component


def linear_warmup(current_step, warmup_steps, lr):
    if current_step >= warmup_steps:
        return lr
    else:
        return current_step / warmup_steps * lr


def init_averages():
    loss_running = deque(maxlen=_RUNNING_AVERAGE_SMALL)
    recall_running_1 = deque(maxlen=_RUNNING_AVERAGE_SMALL)
    recall_running_10 = deque(maxlen=_RUNNING_AVERAGE_SMALL)
    loss_running_big = deque(maxlen=_RUNNING_AVERAGE_BIG)
    recall_running_1_big = deque(maxlen=_RUNNING_AVERAGE_BIG)
    recall_running_10_big = deque(maxlen=_RUNNING_AVERAGE_BIG)
    return (
        loss_running,
        recall_running_1,
        recall_running_10,
        loss_running_big,
        recall_running_1_big,
        recall_running_10_big,
    )


def wrapper_contains_additional_variables(TYPE):
    return "multiplier" in TYPE or "gillick_loss" in TYPE


def should_multiply_logits_outside(TYPE):
    return not ("multiplier" in TYPE or "gillick_loss" in TYPE)


def save_non_final_model(wrapper, save_information: SaveInformation):
    def construct_non_final_name():
        return f"{save_information.output_path}/{wandb.run.name}_{save_information.epoch}_{save_information.recall}.pth"

    name = construct_non_final_name()

    if wrapper_contains_additional_variables(save_information.type):
        torch.save(wrapper.state_dict(), name)
    else:
        torch.save(wrapper.model.state_dict(), name)


def save_final_model(wrapper, save_information: SaveInformation):
    if wrapper_contains_additional_variables(save_information.type):
        torch.save(wrapper.state_dict(), f"{save_information.output_path}/final.pth")
    else:
        torch.save(
            wrapper.model.state_dict(), f"{save_information.output_path}/final.pth"
        )


def save_model(wrapper, save_information: SaveInformation):
    if save_information.is_final:
        save_final_model(wrapper, save_information)
    else:
        save_non_final_model(wrapper, save_information)


def get_wrapper(model, TYPE, LOGIT_MULTIPLIER):
    if "multiplier" in TYPE or "gillick_loss" in TYPE:
        if "gillick_loss" in TYPE:
            return FinetuningWrapperGillick(model, LOGIT_MULTIPLIER, 1, 0)
        return FinetuningWrapperWithMultiplier(model, LOGIT_MULTIPLIER)
    return FinetuningWrapper(model)


def create_model(TYPE, foundation_model, LOGIT_MULTIPLIER=1, MODEL_PATH=None):
    if wrapper_contains_additional_variables(TYPE):
        model = create_additional_variables_model(
            TYPE, foundation_model, LOGIT_MULTIPLIER, MODEL_PATH
        )
    else:
        model = create_default_model(foundation_model, MODEL_PATH)

    return model


def create_additional_variables_model(
    TYPE, foundation_model, LOGIT_MULTIPLIER=1, MODEL_PATH=None
):
    if "gillick_loss" in TYPE:
        # good defaults
        sigmoid_multiplier = 1
        sigmoid_offset = 0
        model = FinetuningWrapperGillick(
            deepcopy(foundation_model),
            LOGIT_MULTIPLIER,
            sigmoid_multiplier,
            sigmoid_offset,
        )
    else:
        model = FinetuningWrapperWithMultiplier(
            deepcopy(foundation_model), LOGIT_MULTIPLIER
        )

    if MODEL_PATH:
        model.load_state_dict(torch.load(MODEL_PATH))

    return model


def create_default_model(foundation_model, MODEL_PATH=None):
    if MODEL_PATH:
        foundation_model.load_state_dict(torch.load(MODEL_PATH))

    return FinetuningWrapper(deepcopy(foundation_model))


# Training ===========================================
@ensure_datatypes(
    [
        Path,
        str,
        int,
        int,
        float,
        str,
        str,
        # Path,
    ],
    {},
)
def train(
    DATASET_DIR: Path,
    FOUNDATION_MODEL_PATH: str,
    EPOCHS: int,
    LOGIT_MULTIPLIER: int,
    LR: float,
    TYPE: str = "entity_names",
    MODEL_SAVE_DIR: str = "models",
    MODEL_PATH: Path = None,
    decay: float = 0.998,
):
    # print all params
    print("BATCH_DIR:", DATASET_DIR)
    print("MODEL_NAME:", FOUNDATION_MODEL_PATH)
    print("EPOCHS:", EPOCHS)
    print("LOGIT_MULTIPLIER:", LOGIT_MULTIPLIER)
    print("TYPE:", TYPE)
    print("LR:", LR)
    print("STATE_DICT_PATH:", MODEL_PATH)

    foundation_model = BertModel.from_pretrained(FOUNDATION_MODEL_PATH)
    if "mentions" in TYPE:
        tokenizer = BertTokenizerFast.from_pretrained(FOUNDATION_MODEL_PATH)
        tokenizer.add_special_tokens({"cls_token": "[M]"})
        foundation_model.resize_token_embeddings(len(tokenizer))

    criterion = nn.CrossEntropyLoss()
    if "gillick_loss" in TYPE:
        print("USING GILLICK LOSS")
        criterion = gillick_loss

    print("Number of tokens in tokenizer:", len(tokenizer))

    model = create_model(TYPE, foundation_model, LOGIT_MULTIPLIER, MODEL_PATH)

    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=LR)

    if "decay" in TYPE:
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, decay)

    running_averages = RunningAverages(_RUNNING_AVERAGE_SMALL, _RUNNING_AVERAGE_BIG)

    print("Starting training")
    for epoch in range(EPOCHS):
        model.to(device)
        model.train()

        train_loss = 0

        batches = load_epoch_blosc(DATASET_DIR, epoch)
        print(f"Loaded {len(batches)} batches")
        epoch_steps = len(batches)

        print("EPOCH:", epoch)

        for batch in tqdm(batches):

            batch_embs, batch_frenemies, labels = batch

            batch_embs = torch.tensor(batch_embs, dtype=torch.int32)
            batch_frenemies = torch.tensor(batch_frenemies, dtype=torch.int32)

            batch_embs = batch_embs.to(device)
            batch_frenemies = batch_frenemies.to(device)

            outputs = model(batch_embs, batch_frenemies)

            if should_multiply_logits_outside(TYPE):
                outputs = outputs * LOGIT_MULTIPLIER

            labels = labels.to(device)
            if "gillick_loss" in TYPE:
                loss = criterion(outputs, labels, model)
            else:
                loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            train_loss += loss.item()

            r_at_1 = batch_recall(outputs, labels, k=1)
            r_at_10 = torch.tensor(0)
            if len(outputs[0]) >= 10:  # if batch is too small, we can't calculate r@10
                r_at_10 = batch_recall(outputs, labels, k=10)

            running_averages.update_loss(loss.item())
            running_averages.update_recall(r_at_1.item(), r_at_10.item())

            wand_dict = {
                "loss": loss.item(),
                "r_at_1": r_at_1.item(),
                "r_at_10": r_at_10.item(),
                "running_loss": running_averages.loss,
                "running_r_at_1": running_averages.recall_1,
                "running_r_at_10": running_averages.recall_10,
                "running_loss_big": running_averages.loss_big,
                "running_r_at_1_big": running_averages.recall_1_big,
                "running_r_at_10_big": running_averages.recall_10_big,
            }
            if "gillick_loss" in TYPE:
                wand_dict["softmax_multiplier"] = model.softmax_multiplier.item()
                wand_dict["sigmoid_multiplier"] = model.sigmoid_multiplier.item()
                wand_dict["sigmoid_offset"] = model.sigmoid_offset.item()

            wandb.log(
                wand_dict,
            )
        print(f"Train loss: {train_loss / epoch_steps}")

        if "decay" in TYPE:
            scheduler.step()

        model.to("cpu")
        if epoch % 50 == 0:
            save_information = SaveInformation(
                TYPE,
                MODEL_SAVE_DIR,
                False,
                epoch,
                wand_dict["running_r_at_1_big"],
            )
            save_model(model, save_information)

    save_information = SaveInformation(TYPE, MODEL_SAVE_DIR, True)
    save_model(model, save_information)


if __name__ == "__main__":
    Fire(train)
