from pathlib import Path
import sys

from utils.multifile_dataset import MultiFileDataset

sys.stdout.reconfigure(line_buffering=True, write_through=True)

from fire import Fire
import numpy as np
import torch
from transformers import BertModel
from tqdm import tqdm

# from data_processors.index.token_index import TokenIndex
from models.batch_sampler import BatchSampler
from models.searcher import ScaNNSearcher
from utils.loaders import get_emb_state_dict, load_embs_and_qids
from finetunings.generate_epochs.datasets import (
    TokensIterableDataset,
    DamuelNeighborsIterator,
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
    LINKS_EMBS_DIR: Path,
    INDEX_TOKENS_DIR: Path,
    INDEX_EMBS_QIDS_DIR: str,
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
    LINKS_EMBS_DIR = Path(LINKS_EMBS_DIR)
    INDEX_TOKENS_DIR = Path(INDEX_TOKENS_DIR)
    OUTPUT_DIR = Path(OUTPUT_DIR)
    STATE_DICT_PATH = Path(STATE_DICT_PATH) if STATE_DICT_PATH else None

    # make sure the output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # print all params
    print("LINKS_EMBS_DIR:", LINKS_EMBS_DIR)
    print("INDEX_TOKENS_DIR:", INDEX_TOKENS_DIR)
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
    TYPE = "mentions"

    if STATE_DICT_PATH:
        state_dict = get_emb_state_dict(STATE_DICT_PATH)
        model.load_state_dict(state_dict)

    model.to(device)

    # token_index = TokenIndex.from_saved(TOKENS_INDEX_DIR)
    index_embs, index_qids = load_embs_and_qids(INDEX_EMBS_QIDS_DIR)
    batch_sampler = BatchSampler(index_embs, index_qids, ScaNNSearcher)

    multifile_dataset = MultiFileDataset(INDEX_TOKENS_DIR)
    tokens = np.array([x[0] for x in multifile_dataset])

    dataset = TokensIterableDataset(LINKS_EMBS_DIR, set(batch_sampler.qids))
    damuel_neighbors_iterator = DamuelNeighborsIterator(
        dataset,
        BATCH_SIZE,
        NEG,
        batch_sampler,
        tokens,
        CONTEXT_SIZE,
    )

    gen = iter(damuel_neighbors_iterator)

    for epoch in range(EPOCHS):
        epoch_steps_counter = 0

        X, lines, Y = None, None, None

        for i, (x, line, y) in tqdm(enumerate(gen), total=STEPS_PER_EPOCH):
            if i == 0:
                X = np.empty((STEPS_PER_EPOCH, *x.shape), dtype=np.int32)
                lines = np.empty((STEPS_PER_EPOCH, *line.shape), dtype=np.int32)
                Y = np.empty((STEPS_PER_EPOCH, *y.shape), dtype=np.int32)
            X[i] = x
            lines[i] = line
            Y[i] = y
            epoch_steps_counter += 1
            if epoch_steps_counter == STEPS_PER_EPOCH:
                epoch_steps_counter = 0
                break
        print(f"Epoch {epoch} created")
        print("Saving")

        # save compressed with lzma and pickle
        np.savez_compressed(
            OUTPUT_DIR / f"epoch_{epoch}.npz",
            X=np.array(X),
            lines=np.array(lines),
            Y=np.array(Y),
        )

        print("Saved")


if __name__ == "__main__":
    Fire(generate)
