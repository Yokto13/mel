import json
from pathlib import Path
from typing import Callable

from data_processors.tokens.damuel.damuel_iterator import DamuelIterator
from data_processors.tokens.damuel.descriptions.entry_processor import EntryProcessor
from data_processors.tokens.tokenizer_wrapper import TokenizerWrapper


class DamuelDescriptionsTokensIteratorBoth(DamuelIterator):
    def __init__(
        self,
        damuel_path,
        tokenizer,
        expected_size=64,
        filename_is_ok: Callable[[str], bool] = None,
    ):
        super().__init__(
            damuel_path,
            tokenizer,
            expected_size,
            filename_is_ok,
        )

        self.entry_processor = EntryProcessor(
            TokenizerWrapper(tokenizer, expected_size),
        )

    def _iterate_file(self, f):
        for line in f:
            damuel_entry = json.loads(line)
            result = self.entry_processor.process_both(damuel_entry)
            if result is not None:
                yield result
