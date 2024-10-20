import json
from pathlib import Path

import pandas as pd

from fire import Fire

mewsli_dir = Path("/home/farhand/bc/data/mewsli/mewsli-9/output/dataset/")

damuel_qids = None

mewsli_langs = ["ar", "de", "en", "es", "fa", "ja", "sr", "ta", "tr"]


def get_mewsli_lang_qids(lang):
    mewsli_path = mewsli_dir / lang / "mentions.tsv"
    data = pd.read_csv(mewsli_path, sep="\t")
    mewsli_qids = data["qid"].unique()
    mewsli_qids = [int(qid[1:]) for qid in mewsli_qids]
    return mewsli_qids


def qid_intersection(lang):
    mewsli_qids = get_mewsli_lang_qids(lang)
    intersection = damuel_qids & set(mewsli_qids)
    print(f"Mewsli {lang} has {len(mewsli_qids)} qids")
    print(f"Intersection of DAMUEL and MEWSLI {lang} is {len(intersection)}")
    print(
        f"Which means that {round(len(intersection) / len(mewsli_qids) * 100, 1)}% of MEWSLI {lang} qids are in DAMUEL"
    )


def qid_intersection_tabular(lang):
    mewsli_qids = get_mewsli_lang_qids(lang)
    intersection = damuel_qids & set(mewsli_qids)
    print(f"{lang} & {round(len(intersection) / len(mewsli_qids) * 100, 1)}\\\\")


def qid_intersection_all(k=5):
    all_mewsli_qids = set()
    for lang in mewsli_langs:
        all_mewsli_qids |= set(get_mewsli_lang_qids(lang))

    intersection = damuel_qids & all_mewsli_qids

    print(f"Damuel contains {len(damuel_qids)} QIDs")

    print(f"Intersection of DAMUEL and MEWSLI is {len(intersection)}")
    print(
        f"Which means that {round(len(intersection) / len(all_mewsli_qids) * 100, 1)}% of MEWSLI qids are in DAMUEL"
    )
    print(
        f"Which means that {round(len(intersection) / len(all_mewsli_qids) * 100, 10)}% of MEWSLI qids are in DAMUEL"
    )

    only_in_mewsli = all_mewsli_qids - damuel_qids

    print(f"First {k} QIDs only in MEWSLI: {sorted(list(only_in_mewsli))[:k]}")


def run(lang):
    global damuel_qids
    damuel_qids = set(x[0] for x in json.load(open(f"damuel_qids_{lang}.json")))
    for lang in mewsli_langs:
        # qid_intersection(lang)
        qid_intersection_tabular(lang)
    qid_intersection_all()


if __name__ == "__main__":
    Fire(run)
