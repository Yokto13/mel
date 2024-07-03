import json
from pathlib import Path
from typing import Callable

from utils.extractors.damuel.damuel_iterator import DamuelIterator
from utils.extractors.damuel.descriptions.entry_processor import EntryProcessor
from data_processors.tokens.tokenizer_wrapper import TokenizerWrapper


class DamuelDescriptionsIterator(DamuelIterator):
    def __init__(
        self,
        damuel_path,
        tokenizer,
        name_token="[M]",
        expected_size=64,
        is_name_ok: Callable[[str], bool] = None,
    ):
        super().__init__(
            damuel_path, tokenizer, expected_size, is_name_ok
        )

        self.tokenizer = tokenizer
        assert self._check_token_in_tokenizer(name_token, tokenizer)

        self.name_token = name_token

        self.entry_processor = EntryProcessor(
            TokenizerWrapper(self.tokenizer, expected_size),
        )

    def _check_token_in_tokenizer(self, token, tokenizer):
        return token in tokenizer.get_vocab()

    def _iterate_file(self, f):
        for line in f:
            damuel_entry = json.loads(line)
            result = self.entry_processor(damuel_entry, self.name_token)
            if result is not None:
                yield result
