from src.utils.extractors.damuel.parser import DamuelParser

class DescriptionTokenizer:
    class MissingLabelException(Exception):
        pass

    def __init__(self, tokenizer_wrapper):
        self.tokenizer_wrapper = tokenizer_wrapper

    def __call__(self, description_damuel_entry: dict, label_token: str = None) -> tuple:
        try: 
            label, description, qid = self._get_data(description_damuel_entry)
        except self.MissingLabelException:
            return None

        label = self._wrap_label_if_requested(label, label_token)

        text = self._construct_text_from_label_and_description(label, description)

        return (self._get_tokens(text), qid)

    def _get_tokens(self, text):
        return self.tokenizer_wrapper.tokenize(text)["input_ids"][0]

    def _get_data(self, description_damuel_entry):
        label = DamuelParser.parse_label(description_damuel_entry)
        description = DamuelParser.parse_description(description_damuel_entry)
        qid = DamuelParser.parse_qid(description_damuel_entry)

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


