import functools

import gin
import numpy as np


@functools.cache
def _load_filter(path: str) -> set:
    """Load QIDs to filter from a `.npy` file."""
    try:
        # TODO: qid remap this loading
        print("Loading filter from", path)
        arr = np.load(path)
    except Exception:
        raise FileNotFoundError(f"Cannot load filter file: {path}")
    return set(arr.tolist())


@gin.configurable
def qid_filter(qids_index: int | None, filter_path: str | None = None):
    """Decorator that filters out QIDs listed in a `.npy` file.

    Args:
        qids_index (int | None): Index of the QIDs in the input data. If None assumes the input data to be just qids.
        filter_path (str): Path to the `.npy` file containing QIDs to filter out.
            Can be empty. If empty, no filtering is applied; the decorator is an identity.
    """

    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            result = fn(*args, **kwargs)
            if filter_path is None:
                return result

            mask_set = set(_load_filter(filter_path))
            is_valid_tuple = isinstance(result, tuple) and 0 <= qids_index < len(result)
            if is_valid_tuple:
                qids = result[qids_index]
            else:
                qids = result
            keep = np.array([q not in mask_set for q in qids])

            if is_valid_tuple:
                updated_result = tuple(result_array[keep] for result_array in result)
            elif qids_index is None and not isinstance(result, tuple):
                updated_result = result[keep]
            else:
                raise ValueError(
                    f"Invalid qids_index {qids_index} for the returned tuple."
                )
            return updated_result

        return wrapper

    return decorator
