import json
from pathlib import Path

from fire import Fire
import pandas as pd

mewsli_dir = Path("/home/farhand/bc/data/mewsli/mewsli-9/output/dataset/")

damuel_qids = None

mewsli_langs = ["ar", "de", "en", "es", "fa", "ja", "sr", "ta", "tr"]


def get_mewsli_lang_qids(lang):
    mewsli_path = mewsli_dir / lang / "mentions.tsv"
    data = pd.read_csv(mewsli_path, sep="\t")
    mewsli_qids = data["qid"]
    mewsli_qids = [int(qid[1:]) for qid in mewsli_qids]
    return mewsli_qids


def qid_intersection_tabular(lang):
    mewsli_qids = get_mewsli_lang_qids(lang)
    intersection = sum(q in damuel_qids for q in mewsli_qids)
    print(f"{lang} & {round(intersection / len(mewsli_qids) * 100, 1)}\\\\")


def qid_intersection_all(k=5):
    all_mewsli_qids = []
    for lang in mewsli_langs:
        all_mewsli_qids.extend(get_mewsli_lang_qids(lang))

    intersection = sum(q in damuel_qids for q in all_mewsli_qids)

    print(f"Damuel contains {len(damuel_qids)} QIDs")

    print(f"Intersection of DAMUEL and MEWSLI is {intersection}")
    print(
        f"Which means that {round(intersection / len(all_mewsli_qids) * 100, 1)}% of MEWSLI qids are in DAMUEL"
    )
    print(
        f"Which means that {round(intersection / len(all_mewsli_qids) * 100, 10)}% of MEWSLI qids are in DAMUEL"
    )


def run(lang):
    global damuel_qids
    damuel_qids = set(x[0] for x in json.load(open(f"damuel_qids_{lang}.json")))
    for lang in mewsli_langs:
        # qid_intersection(lang)
        qid_intersection_tabular(lang)
    qid_intersection_all()


if __name__ == "__main__":
    Fire(run)
