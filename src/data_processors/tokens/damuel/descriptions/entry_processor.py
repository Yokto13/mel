from data_processors.tokens.mention_qid_pair import MentionQidPair


class EntryProcessor:
    def __init__(self, tokenizer_wrapper, qid_parser):
        self.tokenizer_wrapper = tokenizer_wrapper
        self.qid_parser = qid_parser

    def process_both(self, damuel_entry: dict) -> tuple:
        label = self.extract_label(damuel_entry)
        description = self.extract_description(damuel_entry)

        if label is None:
            return None
        if description is None:
            description = ""

        qid = self.qid_parser(damuel_entry["qid"])

        label_tokens = self.tokenizer_wrapper.tokenize(label)
        description_tokens = self.tokenizer_wrapper.tokenize(description)

        return (
            MentionQidPair(label_tokens, qid),
            MentionQidPair(description_tokens, qid),
        )

    def process_to_one(self, damuel_entry: dict, label_token: str = None) -> tuple:
        label = self.extract_label(damuel_entry)
        description = self.extract_description(damuel_entry)

        if label is None:
            return None
        if description is None:
            description = ""

        if label_token is not None:
            label = self._wrap_label(label, label_token)

        text = self._construct_text_from_label_and_description(label, description)

        qid = self.qid_parser(damuel_entry["qid"])

        return MentionQidPair(self.tokenizer_wrapper.tokenize(text), qid)

    def extract_description(self, damuel_entry):
        if "wiki" in damuel_entry:
            return damuel_entry["wiki"]["text"]
        elif "description" in damuel_entry:
            return damuel_entry["description"]
        return None

    def extract_label(self, damuel_entry):
        if "label" in damuel_entry:
            return damuel_entry["label"]
        elif "wiki" in damuel_entry:
            return damuel_entry["wiki"]["title"]
        return None

    def _construct_text_from_label_and_description(self, label, description):
        return f"{label} {description}"

    def _wrap_label(self, label, label_token):
        return f"{label_token}{label}{label_token}"
