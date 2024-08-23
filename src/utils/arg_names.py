import inspect


def get_args_names(f: object) -> list[str]:
    return inspect.getfullargspec(f)[0]
