import functools

import numpy as np

from .loaders import load_qids_npy


@functools.cache
def _load_filter(path: str) -> set:
    """Load QIDs to filter from a `.npy` file."""
    try:
        arr = load_qids_npy(path)
    except Exception:
        raise FileNotFoundError(f"Cannot load filter file: {path}")
    return set(arr.tolist())


def qid_filter(qids_index: int | None, filter_path: str = None):
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

            mask_set = _load_filter(filter_path)
            is_valid_tuple = isinstance(result, tuple) and 0 <= qids_index < len(result)
            if is_valid_tuple:
                qids = result[qids_index]
            else:
                qids = result
            keep = ~np.isin(qids, list(mask_set))

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
