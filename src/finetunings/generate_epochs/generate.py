from pathlib import Path
import pickle
import sys
from time import time

sys.stdout.reconfigure(line_buffering=True, write_through=True)

import blosc
from fire import Fire
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import BertModel, BertTokenizerFast
from tqdm import tqdm

from data_processors.index.token_index import TokenIndex
from utils.loaders import get_emb_state_dict
from finetunings.generate_epochs.datasets import (
    TokensIterableDataset,
    TokensEmbsIterableDataset,
    StatefulIterableDataset,
    DamuelNeighborsIterableDataset,
)

# Settings ===========================================


if torch.cuda.is_available():
    print("CUDA is available!")
    device = torch.device("cuda")
else:
    print("CUDA is not available.")
    device = torch.device("cpu")

SEED = 0
torch.manual_seed(SEED)


def generate(
    TOKENS_DIR: Path,
    TOKENS_INDEX_DIR: Path,
    OUTPUT_DIR: Path,
    MODEL_PATH: str,
    BATCH_SIZE: int,
    EPOCHS: int,
    STEPS_PER_EPOCH: int,
    POS: int,
    NEG: int,
    CONTEXT_SIZE: int,
    TYPE: str = "entity_names",
    STATE_DICT_PATH: str = None,
):
    TOKENS_DIR = Path(TOKENS_DIR)
    TOKENS_INDEX_DIR = Path(TOKENS_INDEX_DIR)
    OUTPUT_DIR = Path(OUTPUT_DIR)
    STATE_DICT_PATH = Path(STATE_DICT_PATH) if STATE_DICT_PATH else None

    # make sure the output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # print all params
    print("TOKENS_DIR:", TOKENS_DIR)
    print("TOKENS_INDEX_DIR:", TOKENS_INDEX_DIR)
    print("OUTPUT_DIR:", OUTPUT_DIR)
    print("MODEL_NAME:", MODEL_PATH)
    print("BATCH_SIZE:", BATCH_SIZE)
    print("EPOCHS:", EPOCHS)
    print("STEPS_PER_EPOCH:", STEPS_PER_EPOCH)
    print("POS:", POS)
    print("NEG:", NEG)
    print("CONTEXT_SIZE:", CONTEXT_SIZE)
    print("TYPE:", TYPE)
    print("STATE_DICT_PATH:", STATE_DICT_PATH)

    model = BertModel.from_pretrained(MODEL_PATH)
    if "mentions" in TYPE:
        tokenizer = BertTokenizerFast.from_pretrained(MODEL_PATH)
        tokenizer.add_special_tokens({"cls_token": "[M]"})
        model.resize_token_embeddings(len(tokenizer))
        TYPE = "mentions"

    if STATE_DICT_PATH:
        state_dict = get_emb_state_dict(STATE_DICT_PATH)
        model.load_state_dict(state_dict)

    model.to(device)

    token_index = TokenIndex.from_saved(TOKENS_INDEX_DIR)

    tokens_dataset = TokensIterableDataset(TOKENS_DIR, TYPE)
    embs_dataset = TokensEmbsIterableDataset(tokens_dataset, model, device)
    stateful_dataset = StatefulIterableDataset(embs_dataset)
    damuel_neighbors_iterable_dataset = DamuelNeighborsIterableDataset(
        token_index, stateful_dataset, BATCH_SIZE, CONTEXT_SIZE, POS, NEG, model, device
    )

    data_loader = DataLoader(
        damuel_neighbors_iterable_dataset,
        batch_size=1,
    )

    for epoch in range(EPOCHS):
        epoch_steps_counter = 0

        batches = []

        for batch in tqdm(data_loader, total=STEPS_PER_EPOCH):
            batch = [x[0] for x in batch]

            batches.append(batch)

            epoch_steps_counter += 1
            if epoch_steps_counter == STEPS_PER_EPOCH:
                epoch_steps_counter = 0
                break
        print(f"Epoch {epoch} created")
        print("Saving")

        # save compressed with lzma and pickle
        with open(OUTPUT_DIR / f"epoch_{epoch}.dat", "wb") as f:
            data = pickle.dumps(batches)
            data = blosc.compress(data)
            f.write(data)

        print("Saved")


if __name__ == "__main__":
    Fire(generate)
