import sys

sys.stdout.reconfigure(line_buffering=True, write_through=True)
from collections import defaultdict, Counter
from pathlib import Path
import wandb

import fire
from utils.loaders import AliasTableLoader

damuels = [
    "de",
    "id",
    "ja",
    "wo",
    "ru",
    "vi",
    "hy",
    "fa",
    "he",
    "uk",
    "pl",
    "sr",
    "zh",
    "cs",
    "ga",
    "nn",
    "ca",
    "te",
    "bg",
    "fr",
    "ro",
    "el",
    "lv",
    "fi",
    "ko",
    "hi",
    "ug",
    "ur",
    "hr",
    "es",
    "af",
    "hu",
    "et",
    "it",
    "sv",
    "gl",
    "mt",
    "pt",
    "ta",
    "eu",
    "ar",
    "la",
    "lt",
    "be",
    "en",
    "sl",
    "sk",
    "se",
    "nl",
    "mr",
    "da",
    "gd",
    "tr",
]
assert len(damuels) == 53

# damuels = ["fa"]
mewslis = ["ar", "de", "en", "es", "fa", "ja", "sr", "ta", "tr"]


def all_languages(
    damuel: str, mewsli: str, only_wiki_links: bool, R: int, lowercase=False
):
    wandb.init(
        project="alias-table-all-languages",
        config={
            "damuel": damuel,
            "mewsli": mewsli,
            "only_wiki_links": only_wiki_links,
            "R": R,
            "lowercase": lowercase,
        },
    )
    damuel = Path(damuel)
    mewsli = Path(mewsli)
    print(f"Args damuel: {damuel}")
    print(f"Args mewsli: {mewsli}")
    print(f"Args only_wiki_links: {only_wiki_links}")
    print(f"Args R: {R}")
    print("Loading DAMUEL...")

    mention_qid_to_count = defaultdict(Counter)

    loader = AliasTableLoader(mewsli, damuel, lowercase=lowercase)

    # Just to test that all mewslis are loadable
    # Lol
    for mewsli_lang in mewslis:
        print(f"Loading MEWSLI {mewsli_lang}...")
        mewsli_mentions_names, mewsli_qids = loader.load_mewsli(mewsli_lang)

    total_dam_len = 0
    for suffix in damuels:
        dam_mentions_names, dam_qids = loader.load_damuel(suffix)
        print(f"Loaded DAMUEL {suffix} with {len(dam_mentions_names)} mentions")
        total_dam_len += len(dam_mentions_names)
        for mention, qid in zip(dam_mentions_names, dam_qids):
            mention_qid_to_count[mention][qid] += 1

    print("Total DAMUEL mentions:", total_dam_len)

    knowledge_base = {}
    for mention, qid_to_count in mention_qid_to_count.items():
        knowledge_base[mention] = set(x[0] for x in qid_to_count.most_common(R))

    print("Loading MEWSLI...")
    for mewsli_lang in mewslis:
        print(f"Loading MEWSLI {mewsli_lang}...")
        mewsli_mentions_names, mewsli_qids = loader.load_mewsli(mewsli_lang)

        print("Knowledge base size: %d", len(knowledge_base))

        print("Evaluating...")
        correct = 0
        for mention, qid in zip(mewsli_mentions_names, mewsli_qids):
            if mention in knowledge_base and qid in knowledge_base[mention]:
                correct += 1

        wandb.log({f"{mewsli_lang}_accuracy": correct / len(mewsli_mentions_names)})

        print("Accuracy:", correct / len(mewsli_mentions_names))
        print("=============================================")


if __name__ == "__main__":
    fire.Fire(all_languages)
