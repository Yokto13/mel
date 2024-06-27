import json
from pathlib import Path

import pandas as pd

mewsli_dir = Path("/home/farhand/bc/data/mewsli/mewsli-9/output/dataset/")

damuel_qids = json.load(open("damuel_qids_specs.json"))

wiki_damuel_qids = {qid for qid, specifier in damuel_qids if specifier == "WIKI"}
print(f"Wiki DAMUEL has {len(wiki_damuel_qids)} qids")
wikidata_damuel_qids = {
    qid for qid, specifier in damuel_qids if specifier == "WIKIDATA"
}
print(f"Wikidata DAMUEL has {len(wikidata_damuel_qids)} qids")
unk_damuel_qids = {qid for qid, specifier in damuel_qids if specifier == "UNK"}
print(f"Unknown DAMUEL has {len(unk_damuel_qids)} qids")

mewsli_langs = ["ar", "de", "en", "es", "fa", "ja", "sr", "ta", "tr"]


def get_mewsli_lang_qids(lang):
    mewsli_path = mewsli_dir / lang / "mentions.tsv"
    data = pd.read_csv(mewsli_path, sep="\t")
    mewsli_qids = data["qid"].unique()
    mewsli_qids = [int(qid[1:]) for qid in mewsli_qids]
    return mewsli_qids


def qid_intersection(lang, damuel):
    mewsli_qids = get_mewsli_lang_qids(lang)
    intersection = damuel & set(mewsli_qids)
    print(f"Mewsli {lang} has {len(mewsli_qids)} qids")
    print(f"Intersection of DAMUEL and MEWSLI {lang} is {len(intersection)}")
    print(
        f"Which means that {round(len(intersection) / len(mewsli_qids) * 100, 1)}% of MEWSLI {lang} qids are in DAMUEL"
    )


def qid_intersection_tabular(lang, damuel):
    mewsli_qids = get_mewsli_lang_qids(lang)
    intersection = damuel & set(mewsli_qids)
    print(f"{lang} & {round(len(intersection) / len(mewsli_qids) * 100, 1)}\\\\")


def qid_intersection_all(damuel, k=5):
    all_mewsli_qids = set()
    for lang in mewsli_langs:
        all_mewsli_qids |= set(get_mewsli_lang_qids(lang))

    intersection = damuel & all_mewsli_qids

    print(f"Damuel contains {len(damuel)} QIDs")

    print(f"Intersection of DAMUEL and MEWSLI is {len(intersection)}")
    print(
        f"Which means that {round(len(intersection) / len(all_mewsli_qids) * 100, 1)}% of MEWSLI qids are in DAMUEL"
    )

    only_in_mewsli = all_mewsli_qids - damuel

    print(f"First {k} QIDs only in MEWSLI: {sorted(list(only_in_mewsli))[:k]}")


if __name__ == "__main__":
    for lang in mewsli_langs:
        # qid_intersection(lang)
        for specifier, damuel in [
            ("WIKI", wiki_damuel_qids),
            ("WIKIDATA", wikidata_damuel_qids),
            ("UNK", unk_damuel_qids),
        ]:
            print(f"Specifier: {specifier}")
            qid_intersection_tabular(lang, damuel)

    for specifier, damuel in [
        ("WIKI", wiki_damuel_qids),
        ("WIKIDATA", wikidata_damuel_qids),
        ("UNK", unk_damuel_qids),
    ]:
        print(f"Specifier: {specifier}")
        qid_intersection_all(damuel)
