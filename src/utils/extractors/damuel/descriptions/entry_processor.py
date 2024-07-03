class EntryProcessor:
    class MissingLabelException(Exception):
        pass

    def __init__(self, tokenizer_wrapper):
        self.tokenizer_wrapper = tokenizer_wrapper
        self.parser = _Parser()

    def __call__(self, damuel_entry: dict, label_token: str = None) -> tuple:
        try: 
            label, description, qid = self._get_data(damuel_entry)
        except self.MissingLabelException:
            return None

        label = self._wrap_label_if_requested(label, label_token)

        text = self._construct_text_from_label_and_description(label, description)

        return (self._get_tokens(text), qid)

    def _get_tokens(self, text):
        return self.tokenizer_wrapper.tokenize(text)["input_ids"][0]

    def _get_data(self, damuel_entry):
        label = self.parser.parse_label(damuel_entry)
        description = self.parser.parse_description(damuel_entry)
        qid = self.parser.parse_qid(damuel_entry)

        if label is None:
            raise self.MissingLabelException("Label is missing.")
        if description is None:
            description = ""

        return label, description, qid

    def _wrap_label_if_requested(self, label, label_token):
        if label_token is not None:
            return self._wrap_label(label, label_token)
        return label

    def _construct_text_from_label_and_description(self, label, description):
        return f"{label} {description}"

    def _wrap_label(self, label, label_token):
        return f"{label_token}{label}{label_token}"

class _Parser:
    def parse_description(self, damuel_entry):
        if "wiki" in damuel_entry:
            return damuel_entry["wiki"]["text"]
        elif "description" in damuel_entry:
            return damuel_entry["description"]
        return None

    def parse_label(self, damuel_entry):
        if "label" in damuel_entry:
            return damuel_entry["label"]
        elif "wiki" in damuel_entry:
            return damuel_entry["wiki"]["title"]
        return None

    def parse_qid(self, damuel_entry):
        return int(damuel_entry["qid"][1:])
