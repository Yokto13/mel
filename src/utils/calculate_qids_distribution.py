from collections import defaultdict
from pathlib import Path

import numpy as np

from .loaders import load_qids


def calculate_qids_distribution_from_links(
    links_dir: Path, index_qids: np.ndarray, transform_fn: callable = lambda x: x
) -> np.ndarray:
    qid_to_cnt: dict[int, int] = defaultdict(int)
    index_qids = set(index_qids)
    for file in links_dir.iterdir():
        if not file.suffix == ".npz":
            continue
        qids = load_qids(file)
        for qid in qids:
            if qid not in index_qids:
                continue
            qid_to_cnt[qid] += 1
    for key in qid_to_cnt:
        qid_to_cnt[key] = transform_fn(qid_to_cnt[key])
    print(len(list(qid_to_cnt.keys())))
    print(len(set(qid_to_cnt.keys()) & set(index_qids)))
    qids_observed_cnt = sum(qid_to_cnt.values())
    res = np.array(
        [qid_to_cnt[qid] / qids_observed_cnt for qid in index_qids], dtype=np.float64
    )
    print(sum(res))
    return res
