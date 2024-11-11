from collections import defaultdict
from pathlib import Path

import numpy as np


def calculate_qids_distribution_from_links(
    links_dir: Path, index_qids: np.ndarray, transform_fn: callable = lambda x: x
) -> np.ndarray:
    qid_to_cnt: dict[int, int] = defaultdict(int)
    for file in links_dir.iterdir():
        if not file.suffix == ".npz":
            continue
        d = np.load(file)
        qids = d["qids"]
        for qid in qids:
            qid_to_cnt[qid] += 1
    for key in qid_to_cnt:
        qid_to_cnt[key] = transform_fn(qid_to_cnt[key])
    qids_observed_cnt = sum(qid_to_cnt.values())
    return np.array([qid_to_cnt[qid] / qids_observed_cnt for qid in index_qids])
