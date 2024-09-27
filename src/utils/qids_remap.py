import numpy as np
from pathlib import Path

import orjson


def _load_json_file(filepath: str | Path) -> dict:
    with open(filepath, "rb") as f:
        return orjson.loads(f.read())


def _convert_qid_keys_to_int(qid_map: dict) -> dict[int, int]:
    return {int(k[1:]): int(v[1:]) for k, v in qid_map.items()}


def load_qids_remap(filepath: str | Path) -> dict[int, int]:
    qid_map = _load_json_file(filepath)
    return _convert_qid_keys_to_int(qid_map)


def qids_remap(qids: np.array, old_to_new_qids_path: str | Path):
    old_to_new_qids = load_qids_remap(old_to_new_qids_path)
    return np.array(
        [q if q not in old_to_new_qids else old_to_new_qids[q] for q in qids]
    )
