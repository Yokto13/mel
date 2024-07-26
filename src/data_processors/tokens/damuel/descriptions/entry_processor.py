import functools


def _should_skip(wrapped):
    """
    Decorator that decides whether method should process entry.

    Expects signature (self, entry).
    If method should not process entry, the method is not call and None is returned.

    This might be unecessary too complex but I did not want to rewrite logic of all the different process_* methods.
    """

    def _should_entry_be_skipped(*args, **kwargs):
        self = args[0]
        entry = args[1]
        assert type(entry) == dict
        if self.only_pages and "wiki" not in entry:
            return True
        return False

    @functools.wraps(wrapped)
    def _wrapper(*args, **kwargs):
        if _should_entry_be_skipped(*args, **kwargs):
            return None
        return wrapped(*args, **kwargs)

    return _wrapper


class EntryProcessor:
    def __init__(self, tokenizer_wrapper, only_pages=False):
        self.tokenizer_wrapper = tokenizer_wrapper
        self.only_pages = only_pages

    @_should_skip
    def process_both(self, damuel_entry: dict) -> tuple:
        label = self.extract_title(damuel_entry)
        description = self.extract_description(damuel_entry)

        if label is None:
            return None
        if description is None:
            description = ""

        qid = int(damuel_entry["qid"][1:])

        label_tokens = self.tokenizer_wrapper.tokenize(label)
        description_tokens = self.tokenizer_wrapper.tokenize(description)

        return (
            (label_tokens, qid),
            (description_tokens, qid),
        )

    @_should_skip
    def process_to_one(self, damuel_entry: dict, label_token: str = None) -> tuple:
        label = self.extract_title(damuel_entry)
        description = self.extract_description(damuel_entry)

        if label is None:
            return None
        if description is None:
            description = ""

        if label_token is not None:
            label = self._wrap_label(label, label_token)

        text = self._construct_text_from_label_and_description(label, description)

        qid = int(damuel_entry["qid"][1:])
        return self.tokenizer_wrapper.tokenize(text), qid

    def extract_description(self, damuel_entry):
        if "wiki" in damuel_entry:
            return damuel_entry["wiki"]["text"]
        elif "description" in damuel_entry:
            return damuel_entry["description"]
        return None

    def extract_title(self, damuel_entry):
        if "wiki" in damuel_entry:
            return damuel_entry["wiki"]["title"]
        elif "label" in damuel_entry:
            return damuel_entry["label"]
        return None

    def _construct_text_from_label_and_description(self, label, description):
        return f"{label} {description}"

    def _wrap_label(self, label, label_token):
        return f"{label_token}{label}{label_token}"
