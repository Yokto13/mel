""" A quick, dirty script that calculates how many disambiguations QIDs appear in Mewsli. """

import json
from pathlib import Path

import pandas as pd
import pywikibot
from pywikibot import exceptions

site = pywikibot.Site("wikidata", "wikidata")

mewsli_dir = Path("/home/farhand/bc/data/mewsli/mewsli-9/output/dataset/")

damuel_qids = json.load(open("damuel_qids.json"))
damuel_qids = set(damuel_qids)

wikidata_url = "https://www.wikidata.org/wiki/"

mewsli_langs = ["ar", "de", "en", "es", "fa", "ja", "sr", "ta", "tr"]


def get_mewsli_lang_qids(lang):
    mewsli_path = mewsli_dir / lang / "mentions.tsv"
    data = pd.read_csv(mewsli_path, sep="\t")
    mewsli_qids = data["qid"].unique()
    mewsli_qids = [int(qid[1:]) for qid in mewsli_qids]
    return mewsli_qids


def is_disambiguation(qid):
    item = pywikibot.ItemPage(site, f"Q{qid}")
    try:
        item.get()
    except exceptions.IsRedirectPageError:
        return False
    except exceptions.NoPageError:
        return False
    # P31 is instance of
    # Q4167410 is the disambiguation page
    return "P31" in item.claims and any(
        claim.getTarget().id == "Q4167410" for claim in item.claims["P31"]
    )


def get_all_mewsli_qids():
    all_mewsli_qids = set()
    for lang in mewsli_langs:
        all_mewsli_qids |= set(get_mewsli_lang_qids(lang))
    return all_mewsli_qids


def get_disambiguations(only_in_mewsli):
    disambiguations = []
    for i, qid in enumerate(only_in_mewsli):
        if is_disambiguation(qid):
            disambiguations.append(qid)
        if i % 1000 == 0:
            print(f"Processed {i} QIDs")
    return disambiguations


def disambiguation_all():
    all_mewsli_qids = get_all_mewsli_qids()
    only_in_mewsli = all_mewsli_qids - damuel_qids
    print("QIDs count only in Mewsli", len(only_in_mewsli))
    disambiguations = get_disambiguations(only_in_mewsli)
    print(f"Disambiguations in MEWSLI: {len(disambiguations)}")
    print(
        f"Which means that {round(len(disambiguations) / len(only_in_mewsli) * 100, 1)}% of MEWSLI qids are disambiguations"
    )


if __name__ == "__main__":
    disambiguation_all()
