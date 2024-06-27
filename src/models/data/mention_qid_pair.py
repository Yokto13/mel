from typing import Mapping, NamedTuple

from transformers import BatchEncoding


class MentionQidPair(NamedTuple):
    tokenization_output: BatchEncoding
    qid: str | int

    def __hash__(self) -> int:
        return self.qid.__hash__() ^ hash(
            tuple(self.tokenization_output["input_ids"][0].reshape(-1))
        )

    def __eq__(self, other):
        if not isinstance(other, Mapping):
            return NotImplemented
        return dict(self.items()) == dict(other.items()).all()
