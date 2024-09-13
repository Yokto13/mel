import os
from tqdm import tqdm

import numpy as np

from utils.loaders import load_embs_and_qids


def sum_and_normalize_embs(embs_list):
    sum_embs = np.sum(embs_list, axis=0)
    norm = np.linalg.norm(sum_embs)
    return sum_embs / norm


def combine_embs_by_qid(input_dir_path: str, output_dir_path: str) -> None:
    embs, qids = load_embs_and_qids(input_dir_path)

    qid_to_embs = {}

    for emb, qid in tqdm(
        zip(embs, qids), total=len(qids), desc="Grouping embeddings by qid"
    ):
        if qid not in qid_to_embs:
            qid_to_embs[qid] = []
        qid_to_embs[qid].append(emb)

    unique_qids = np.array(list(qid_to_embs.keys()))
    print(f"Number of unique qids: {len(unique_qids)}")
    print("NUmber of all qids:", len(qids))
    normalized_sum_embs = []
    for embs in tqdm(
        qid_to_embs.values(),
        total=len(qid_to_embs),
        desc="Normalizing and summing embeddings",
    ):
        normalized_sum_embs.append(sum_and_normalize_embs(embs))

    os.makedirs(output_dir_path, exist_ok=True)
    np.savez_compressed(
        os.path.join(output_dir_path, "embs_qids.npz"),
        embs=normalized_sum_embs,
        qids=unique_qids,
    )
