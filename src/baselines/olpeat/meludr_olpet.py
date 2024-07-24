from collections import Counter, defaultdict
from hashlib import sha1
import sys
from pathlib import Path
import numpy as np

import fire
import torch
from torch.utils.data import DataLoader, Dataset
import wandb

sys.stdout.reconfigure(line_buffering=True, write_through=True)

sys.path.append("/home/farhand/bc")

from src.models.index import Index
from src.experiments.helpers.paths_validator import paths_exist


class SimpleDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def get_unique_n(iterable, n):
    seen = set()
    for i in iterable:
        if i not in seen:
            seen.add(i)
            yield i
        if len(seen) == n:
            break


class RecallCalculator:
    def __init__(self, scann_index, qids_in_index) -> None:
        self.scann_index = scann_index
        self.qids_in_index = qids_in_index

    def recall(self, mewsli_embs, mewsli_qids, k: int):
        qid_was_present = self._process_for_recall(mewsli_embs, mewsli_qids, k)
        return self._calculate_recall(qid_was_present)

    def _calculate_recall(self, qid_was_present):
        return sum(qid_was_present) / len(qid_was_present)

    def _get_neighboring_qids(self, queries_embs, k):
        qids_per_query = []
        neighbors, dists = self.scann_index.search_batched(
            queries_embs, final_num_neighbors=max(100000, k)
        )
        for ns in neighbors:
            ns_qids = self.qids_in_index[ns]
            unique_ns_qids = list(get_unique_n(ns_qids, k))
            qids_per_query.append(unique_ns_qids)
        return qids_per_query

    def _process_for_recall(self, mewsli_embs, mewsli_qids, k):
        qid_was_present = []

        for emb, qid in zip(mewsli_embs, mewsli_qids):
            negihboring_qids = self._get_neighboring_qids([emb], k)
            qid_was_present.append(qid in negihboring_qids[0])

        return qid_was_present


def load_embs(dir_path):
    if isinstance(dir_path, str):
        dir_path = Path(dir_path)

    embs_all, qids_all = [], []

    for fname in sorted(dir_path.iterdir()):
        if not fname.name.startswith("embs"):
            continue
        print("Loading", fname)
        embs = np.load(fname)
        qids = np.load(dir_path / f"qids_{fname.stem.split('_')[1]}.npy")

        for emb, qid in zip(embs, qids):
            embs_all.append(emb)
            qids_all.append(qid)
    return embs_all, qids_all


@paths_exist(path_arg_ids=[0, 1])
def main(
    damuel_entities: str,
    mewsli: str,
    R,
    use_scann=False,
    damuel_links: str = None,
    bs=32,
):
    wandb.init(
        project="no-training-entities",
        config={
            "R": R,
            "use_scann": use_scann,
            "damuel_entities": damuel_entities,
            "mewsli": mewsli,
            "damuel_links": damuel_links,
        },
    )

    if damuel_links is not None:
        print("Loading DAMUEL links...")
        damuel_embs, damuel_qids = load_embs(damuel_links)

        print("Loading DAMUEL entities...")
        damuel_embs_entities, damuel_qids_entities = load_embs(damuel_entities)

        damuel_qids = np.concatenate([damuel_qids, damuel_qids_entities])
        del damuel_qids_entities
        damuel_embs = np.concatenate([damuel_embs, damuel_embs_entities])
        del damuel_embs_entities

    else:
        print("Loading DAMUEL entities...")
        damuel_embs, damuel_qids = load_embs(damuel_entities)
        damuel_embs = np.array(damuel_embs)
        damuel_qids = np.array(damuel_qids)

    # normalize
    damuel_embs = damuel_embs / np.linalg.norm(damuel_embs, axis=1, keepdims=True)

    if not damuel_embs.flags["C_CONTIGUOUS"]:
        damuel_embs = np.ascontiguousarray(damuel_embs)

    emb_qid_d = defaultdict(Counter)
    for emb, qid in zip(damuel_embs, damuel_qids):
        emb_qid_d[sha1(emb.tobytes()).hexdigest()][qid] += 1

    # keep only top R qids per emb
    for emb_hash, qid_counter in emb_qid_d.items():
        emb_qid_d[emb_hash] = [qid for qid, _ in qid_counter.most_common(R)]

    R = min(R, len(damuel_qids))

    # filter embs and qids
    new_embs, new_qids = [], []
    for emb, qid in zip(damuel_embs, damuel_qids):
        emb_hash = sha1(emb.tobytes()).hexdigest()
        if qid in emb_qid_d[emb_hash]:
            new_embs.append(emb)
            new_qids.append(qid)

    damuel_embs = np.array(new_embs)
    damuel_qids = np.array(new_qids)

    print("Loading MEWSLI entities...")
    mewsli_embs, mewsli_qids = load_embs(mewsli)
    mewsli_embs = np.array(mewsli_embs)

    # normalize
    mewsli_embs = mewsli_embs / np.linalg.norm(mewsli_embs, axis=1, keepdims=True)

    mewsli_qids = np.array(mewsli_qids)

    if use_scann:
        print(damuel_embs.shape, damuel_qids.shape)
        scann_index = Index(damuel_embs, damuel_qids).scann_index
        rc = RecallCalculator(scann_index, damuel_qids)

        print("Calculating recall...")
        recall = rc.recall(mewsli_embs, mewsli_qids, R)
        wandb.log({"recall": recall})
        print("Recall:", recall)
    else:
        print("Calculating recall...")

        damuel_embs = torch.tensor(damuel_embs, dtype=torch.float16).float()
        mewsli_embs = torch.tensor(mewsli_embs, dtype=torch.float16).float()

        device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        damuel_embs = damuel_embs.to(device)

        print("Calculating similarities...")
        print(damuel_embs.shape, mewsli_embs.shape)

        mewsli = SimpleDataset(mewsli_embs)
        mewsli_dataloader = DataLoader(mewsli, batch_size=bs, shuffle=False)

        correct = 0
        j = 0
        for batch in mewsli_dataloader:
            print("Processing batch")
            batch = batch.to(device)
            similarities = torch.matmul(batch, damuel_embs.T)
            similarities = similarities.cpu().numpy()
            for i, mewsli_qid in enumerate(mewsli_qids[j : j + bs]):
                if mewsli_qid in damuel_qids[np.argsort(similarities[i])[-R:]]:
                    correct += 1
            j += bs
        wandb.log({"recall": correct / len(mewsli_qids)})
        print("Recall:", correct / len(mewsli_qids))


if __name__ == "__main__":
    fire.Fire(main)
