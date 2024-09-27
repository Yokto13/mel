from functools import wraps
from itertools import zip_longest
from pathlib import Path


def paths_exist(path_arg_ids: list):
    """Controlls that a path exists before calling the function

    path_arg_ids: list of indexes of the arguments that are valid paths.
    """

    def _paths_exist(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            for i in path_arg_ids:
                path_candidate = args[i]
                if isinstance(path_candidate, str):
                    path_candidate = Path(path_candidate)
                if not path_candidate.exists():
                    raise FileNotFoundError(f"Path {path_candidate} does not exist")
            return f(*args, **kwargs)

        return wrapper

    return _paths_exist


def ensure_datatypes(args_constructors: list, kwargs_constructors: dict):
    """Applys constructor functions to the arguments before calling the function"""

    def _ensure_datatypes(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            (*args,) = (
                constructor(arg)
                for arg, constructor in zip_longest(
                    args, args_constructors[: len(args)], fillvalue=lambda x: x
                )
            )
            for key, constructor in kwargs_constructors.items():
                if key in kwargs:
                    kwargs[key] = constructor(kwargs[key])
            return f(*args, **kwargs)

        return wrapper

    return _ensure_datatypes
