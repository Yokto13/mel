import lzma
from math import inf
import os
from pathlib import Path
from functools import wraps
from time import time
import pickle
import sys
import fire
import numpy as np
import torch

from torch.utils.data import Dataset, DataLoader

from transformers import BertModel, BertTokenizerFast

# print work dir
print(os.getcwd())

from models.data.mention_qid_pair import MentionQidPair
from utils.loaders import get_emb_state_dict

BATCH_SIZE = 1024

if torch.cuda.is_available():
    print("CUDA is available!")
    device = torch.device("cuda")
else:
    print("CUDA is not available.")
    device = torch.device("cpu")


class TokensDataset(Dataset):
    def __init__(self, tokens_list: list[MentionQidPair]):
        self.tokens_list = tokens_list

    def __len__(self):
        return len(self.tokens_list)

    def __getitem__(self, idx):
        return (
            self.tokens_list[idx].tokenization_output["input_ids"],
            self.tokens_list[idx].tokenization_output["attention_mask"],
            self.tokens_list[idx].qid,
        )


def create_embs_from_pairs_and_model(mentions: list[MentionQidPair], model):
    mentions_dataset = TokensDataset(mentions)

    mentions_loader = DataLoader(mentions_dataset, batch_size=BATCH_SIZE, shuffle=False)

    embs = []
    qids = []

    for mention_input_ids, mention_attention, mention_qids in mentions_loader:
        # drop the penultimate dim
        mention_input_ids = mention_input_ids.squeeze(1)
        mention_attention = mention_attention.squeeze(1)

        mention_input_ids = mention_input_ids.to(device)
        mention_attention = mention_attention.to(device)

        print("mention_input_ids", mention_input_ids.shape)
        print("mention_attention", mention_attention.shape)

        with torch.no_grad():
            embs_batch = model(
                mention_input_ids, attention_mask=mention_attention
            ).pooler_output

        qids.extend(mention_qids)
        embs.extend(embs_batch.cpu().numpy())
    return np.array(embs), np.array(qids)


def tokens_to_embeddings(tokens_dir, embedding_making_f, output_dir):
    print("Starting to load tokens")
    print(os.listdir(tokens_dir))
    print(tokens_dir)
    for fn in sorted(os.listdir(tokens_dir)):
        if not fn.startswith("mentions_"):
            continue
        hash_str = fn.split("_")[1]
        if "." in hash_str:
            hash_str = hash_str.split(".")[0]
        print("Starting loading", fn)
        with open(tokens_dir / fn, "rb") as f:
            mentions = pickle.load(f)

        print("Loaded. Starting to produce embeddings.")
        embs, qids = create_embs_from_pairs_and_model(mentions, embedding_making_f)
        print("Produced embeddings")

        embs = embs.astype(np.float16)

        print("Saving...")
        np.save(output_dir / f"embs_{hash_str}.npy", embs)
        np.save(output_dir / f"qids_{hash_str}.npy", qids)


def generate_embs(tokens_dir, model_name, output_dir, model_state_dict_path=None):
    if isinstance(tokens_dir, str):
        tokens_dir = Path(tokens_dir)
    if isinstance(output_dir, str):
        output_dir = Path(output_dir)
    if model_state_dict_path is not None:
        if isinstance(model_state_dict_path, str):
            model_state_dict_path = Path(model_state_dict_path)

    print("Loading model")

    tokenizer = BertTokenizerFast.from_pretrained(model_name)
    tokenizer.add_special_tokens({"cls_token": "[M]"})

    labse_model = BertModel.from_pretrained(model_name)
    labse_model.resize_token_embeddings(len(tokenizer))
    labse_model.eval()
    labse_model.to(device)

    if model_state_dict_path is not None:
        print("Loading model state dict")
        labse_model.load_state_dict(get_emb_state_dict(model_state_dict_path))

    print("Model loaded. Starting to produce embeddings.")

    tokens_to_embeddings(tokens_dir, labse_model, output_dir)


if __name__ == "__main__":
    fire.Fire(generate_embs)
