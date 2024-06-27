import sys

sys.stdout.reconfigure(line_buffering=True, write_through=True)
from collections import defaultdict, Counter
from math import inf
from pathlib import Path
import wandb

import fire
import numpy as np
from rapidfuzz import process, fuzz
from tqdm import tqdm

from utils.loaders import load_mewsli, load_damuel


def get_batch_gen(aliases, qids, batch_size):
    for i in range(0, len(aliases), batch_size):
        yield aliases[i : i + batch_size], qids[i : i + batch_size]


def filter_qids(qids, aliases, max_per_qid):
    if max_per_qid == inf:
        return aliases, qids
    filter_counter = defaultdict(int)
    wanted_ids = set()
    for i, qid in enumerate(qids):
        if filter_counter[qid] < max_per_qid:
            filter_counter[qid] += 1
            wanted_ids.add(i)
    return [aliases[i] for i in wanted_ids], [qids[i] for i in wanted_ids]


def solve_batch(batch, aliases, qids, R, workers):
    batch_mentions = batch[0]
    batch_qids = np.array(batch[1])
    dists = process.cdist(batch_mentions, aliases, scorer=fuzz.QRatio, workers=workers)
    sorted_indices = np.argsort(-dists, axis=1)
    assert sorted_indices.shape[0] == len(batch_mentions)
    predicted_qids = qids[sorted_indices]
    unique_inds = np.array(
        [
            sorted(np.unique(predicted_qids[i], return_index=True)[1])
            for i in range(predicted_qids.shape[0])
        ]
    )
    rows = np.arange(predicted_qids.shape[0])[:, None]
    predicted_qids_without_dups = predicted_qids[rows, unique_inds]
    assert predicted_qids_without_dups.shape[1] == len(set(qids))
    top_R = predicted_qids_without_dups[:, :R]
    return np.sum(np.any(top_R == batch_qids[:, None], axis=1))


def build_mention_qid_map(dam_mentions_names, dam_qids, R):
    mention_qid_to_count = defaultdict(Counter)
    for mention, qid in zip(dam_mentions_names, dam_qids):
        mention_qid_to_count[mention][qid] += 1
    knowledge_base = {}
    for mention, qid_to_count in mention_qid_to_count.items():
        knowledge_base[mention] = set(x[0] for x in qid_to_count.most_common(R))
    return knowledge_base


def string_similarity(
    damuel,
    mewsli,
    only_wiki_links,
    R,
    workers,
    xz=True,
    batch_size=1000,
    max_per_qid=inf,
    lowercase=False,
):
    wandb.init(
        project="string-similarity-one-language",
        config={
            "damuel": damuel,
            "mewsli": mewsli,
            "only_wiki_links": only_wiki_links,
            "R": R,
            "max_per_qid": max_per_qid,
            "lowercase": lowercase,
        },
    )
    if max_per_qid > 10**20:
        max_per_qid = inf

    damuel = Path(damuel)
    mewsli = Path(mewsli)
    print("Args damuel: %s", damuel)
    print("Args mewsli: %s", mewsli)
    print("Args only_wiki_links: %s", only_wiki_links)
    print("Args R: %s", R)

    print("Loading MEWSLI...")
    mewsli_mentions_names, mewsli_qids = load_mewsli(mewsli, lowercase=lowercase)

    print("Loading DAMUEL...")
    dam_mentions_names, dam_qids = load_damuel(
        damuel, only_wiki_links, use_xz=xz, lowercase=lowercase
    )

    dam_mentions_names, dam_qids = filter_qids(
        dam_qids, dam_mentions_names, max_per_qid
    )

    knowledge_base = build_mention_qid_map(dam_mentions_names, dam_qids, R)

    aliases = []
    qids = []
    for mention in knowledge_base:
        for qid in knowledge_base[mention]:
            aliases.append(mention)
            qids.append(qid)

    qids = np.array(qids)

    # filter exact matches
    filtered_mewsli_mentions_names = []
    filtered_mewsli_qids = []

    correct = 0

    total_eval_len = len(mewsli_mentions_names)

    for mention, qid in zip(mewsli_mentions_names, mewsli_qids):
        if mention in knowledge_base and qid in knowledge_base[mention]:
            correct += 1
        else:
            filtered_mewsli_mentions_names.append(mention)
            filtered_mewsli_qids.append(qid)

    mewsli_mentions_names = filtered_mewsli_mentions_names
    mewsli_qids = np.array(filtered_mewsli_qids)

    print("Starting evaluation...")
    print(mewsli_mentions_names[:10])
    print(mewsli_qids[:10])

    batch_gen = get_batch_gen(mewsli_mentions_names, mewsli_qids, batch_size)
    for batch in tqdm(batch_gen, total=len(mewsli_mentions_names) // batch_size):
        correct += solve_batch(batch, aliases, qids, R, workers)

    print(f"R@{R}", correct / total_eval_len)
    wandb.log({f"R@{R}": correct / total_eval_len})


if __name__ == "__main__":
    fire.Fire(run)
