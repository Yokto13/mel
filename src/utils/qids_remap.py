import functools
from pathlib import Path
from typing import Any, Callable, Tuple

import gin
import numpy as np

import orjson


def _load_json_file(filepath: str | Path) -> dict:
    with open(filepath, "r") as f:
        return orjson.loads(f.read())


def _convert_qid_keys_to_int(qid_map: dict) -> dict[int, int]:
    return {int(k[1:]): int(v[1:]) for k, v in qid_map.items()}


def load_qids_remap(filepath: str | Path) -> dict[int, int]:
    qid_map = _load_json_file(filepath)
    return _convert_qid_keys_to_int(qid_map)


_qids_dict = None


@gin.configurable
def qids_remap(qids: np.array, old_to_new_qids_path: str | Path) -> np.array:
    global _qids_dict
    if _qids_dict is None:
        _qids_dict = load_qids_remap(old_to_new_qids_path)
    remapped_qids = np.array([_qids_dict.get(q, q) for q in qids], dtype=qids.dtype)
    return remapped_qids


@gin.configurable
def remap_qids_decorator(qids_index: int, json_path: str) -> Callable:
    def decorator(
        func: Callable[..., Tuple[Any, ...]]
    ) -> Callable[..., Tuple[Any, ...]]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Tuple[Any, ...]:
            result = func(*args, **kwargs)

            # Check if the result is a tuple and if the qids_index is within bounds
            if isinstance(result, tuple) and 0 <= qids_index < len(result):
                qids = result[qids_index]

                # Check if the qids element is a numpy array
                remapped_qids = qids_remap(qids, json_path)
                updated_result = (
                    result[:qids_index] + (remapped_qids,) + result[qids_index + 1 :]
                )
                return updated_result
            else:
                raise ValueError(
                    f"Invalid qids_index {qids_index} for the returned tuple."
                )

        return wrapper

    return decorator
