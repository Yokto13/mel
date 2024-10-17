import sys

sys.stdout.reconfigure(line_buffering=True, write_through=True)
from collections import Counter, defaultdict
from math import inf
from pathlib import Path

import fire
import spacy

import wandb

from utils.loaders import load_damuel, load_mewsli

lang_model = {
    "de": "de_core_news_sm",
    "en": "en_core_web_sm",
    "es": "es_core_news_sm",
    "ja": "ja_core_news_sm",
}


def alias_table_with_lemmas(
    damuel, mewsli, only_wiki_links, R, xz=False, max_per_qid=inf, lowercase=False
):
    wandb.init(
        project="alias-table-one-language-lemmas",
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

    lang_index_mewsli = -2
    nlp = spacy.load(lang_model[mewsli.parts[lang_index_mewsli]])

    print("Loading MEWSLI...")
    mewsli_mentions_names, mewsli_qids = load_mewsli(mewsli, lowercase=lowercase)
    mewsli_mentions_names = [
        " ".join([y.lemma_ for y in x]) for x in nlp.pipe(mewsli_mentions_names)
    ]

    print("Loading DAMUEL...")
    dam_mentions_names, dam_qids = load_damuel(
        damuel, only_wiki_links, use_xz=xz, lowercase=lowercase
    )

    dam_mentions_names = [
        " ".join([y.lemma_ for y in x]) for x in nlp.pipe(dam_mentions_names)
    ]

    if max_per_qid < inf:
        filter_counter = defaultdict(int)
        wanted_ids = set()
        for i, qid in enumerate(dam_qids):
            if filter_counter[qid] < max_per_qid:
                filter_counter[qid] += 1
                wanted_ids.add(i)
        dam_mentions_names = [dam_mentions_names[i] for i in wanted_ids]
        dam_qids = [dam_qids[i] for i in wanted_ids]

    mention_qid_to_count = defaultdict(Counter)
    for mention, qid in zip(dam_mentions_names, dam_qids):
        mention_qid_to_count[mention][qid] += 1

    knowledge_base = {}
    for mention, qid_to_count in mention_qid_to_count.items():
        knowledge_base[mention] = set(x[0] for x in qid_to_count.most_common(R))

    print("Knowledge base size: %d", len(knowledge_base))

    print("Evaluating...")
    correct = 0
    for mention, qid in zip(mewsli_mentions_names, mewsli_qids):
        if mention in knowledge_base and qid in knowledge_base[mention]:
            correct += 1
        else:
            print(mention)

    print("Accuracy:", correct / len(mewsli_mentions_names))
    wandb.log({"accuracy": correct / len(mewsli_mentions_names)})


if __name__ == "__main__":
    fire.Fire(alias_table_with_lemmas)
