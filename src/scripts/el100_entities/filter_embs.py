""" Filters embeddings and QIDs that should be in the final KB. """

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from fire import Fire
from tqdm import tqdm


def filter_embs(source_dir: Path, dest_dir: Path, wanted: list[int]):
    for fn in tqdm(source_dir.iterdir()):
        if not fn.is_file() or not fn.name.startswith("embs_"):
            continue
        qids_fn = fn.name.replace("embs", "qids")
        data = np.load(source_dir / fn)
        qids = np.load(source_dir / qids_fn)
        mask = np.isin(qids, wanted)

        data = data[mask]
        qids = qids[mask]

        new_fn = dest_dir / fn.name
        new_qids_fn = dest_dir / qids_fn

        np.save(new_fn, data)
        np.save(new_qids_fn, qids)


def load_wanted(path_to_wanted: Path):
    with open(path_to_wanted, "r") as f:
        qid_loc_pair = json.load(f)
    wanted = defaultdict(list)
    for qid, lang in qid_loc_pair.items():
        wanted[lang].append(qid)
    return wanted


def run(dir_with_langs, dest_dir, path_to_wanted):
    dir_with_langs = Path(dir_with_langs)
    dest_dir = Path(dest_dir)
    dir_with_langs = Path(dir_with_langs)
    wanted = load_wanted(path_to_wanted)
    for lang in tqdm(["ar", "de", "en", "fa", "ja", "sr", "ta", "tr"]):
        filter_embs(dir_with_langs / lang / "damuel_embs", dest_dir, wanted[lang])


if __name__ == "__main__":
    Fire(run)
