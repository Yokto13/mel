import functools
import logging
from pathlib import Path
from typing import Any, Callable, Tuple

import gin
import numpy as np

import orjson

_logger = logging.getLogger("utils.qids_remap")


def _load_json_file(filepath: str | Path) -> dict:
    with open(filepath, "r") as f:
        return orjson.loads(f.read())


def _convert_qid_keys_to_int(qid_map: dict) -> dict[int, int]:
    return {int(k[1:]): int(v[1:]) for k, v in qid_map.items()}


def load_qids_remap(filepath: str | Path) -> dict[int, int]:
    qid_map = _load_json_file(filepath)
    return _convert_qid_keys_to_int(qid_map)


_qids_lookup = None
# This should be a safe default for the forseeable future.
# Also I don't want to waste too much memory, this is already like 1 GB for a lookup table
# that maps nearly everything to itself.
_largest_qid = int(3e8)


@gin.configurable
def qids_remap(qids: np.array, old_to_new_qids_path: str | Path) -> np.array:
    global _qids_lookup
    if _qids_lookup is None:
        qids_map = load_qids_remap(old_to_new_qids_path)
        global _largest_qid
        map_largest_qid = max(*qids_map.keys(), *qids_map.values())
        if map_largest_qid < _largest_qid:
            _logger.warning(
                f"""Map largest qid {map_largest_qid} is smaller than the largest qid {_largest_qid}.
                This might mean that the map is not complete or old.
                """
            )
        _largest_qid = max(_largest_qid, map_largest_qid)
        _qids_lookup = np.arange(_largest_qid + 1, dtype=qids.dtype)
        keys = np.fromiter(qids_map.keys(), dtype=_qids_lookup.dtype)
        vals = np.fromiter(qids_map.values(), dtype=_qids_lookup.dtype)
        _qids_lookup[keys] = vals

    remapped_qids = _qids_lookup[qids]
    return remapped_qids


@gin.configurable
def remap_qids_decorator(qids_index: int | None, json_path: str) -> Callable:
    def decorator(
        func: Callable[..., Tuple[Any, ...]],
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
            elif qids_index is None:
                return qids_remap(result, json_path)
            else:
                raise ValueError(
                    f"Invalid qids_index {qids_index} for the returned tuple."
                )

        return wrapper

    return decorator
