import json
from pathlib import Path
from typing import Callable

from data_processors.tokens.damuel.damuel_iterator import DamuelIterator
from data_processors.tokens.damuel.descriptions.entry_processor import EntryProcessor
from data_processors.tokens.tokenizer_wrapper import TokenizerWrapper


class DamuelDescriptionsTokensIteratorFinetuning(DamuelIterator):
    def __init__(
        self,
        damuel_path,
        tokenizer,
        name_token="[M]",
        expected_size=64,
        filename_is_ok: Callable[[str], bool] = None,
    ):
        super().__init__(
            damuel_path,
            tokenizer,
            expected_size,
            filename_is_ok,
        )

        self.name_token = name_token

        assert (
            name_token in tokenizer.get_vocab()
            or name_token.lower() in tokenizer.get_vocab()
        )  # some tokenizers enforce lower case

        self.entry_processor = EntryProcessor(
            TokenizerWrapper(self.tokenizer, expected_size),
        )

    def _iterate_file(self, f):
        for line in f:
            damuel_entry = json.loads(line)
            result = self.entry_processor.process_to_one(damuel_entry, self.name_token)
            if result is not None:
                yield result


class DamuelDescriptionsPagesTokensIteratorFinetuning(
    DamuelDescriptionsTokensIteratorFinetuning
):
    def __init__(
        self,
        damuel_path,
        tokenizer,
        name_token="[M]",
        expected_size=64,
        filename_is_ok: Callable[[str], bool] = None,
    ):
        super().__init__(
            damuel_path, tokenizer, name_token, expected_size, filename_is_ok
        )

        self.entry_processor = EntryProcessor(
            TokenizerWrapper(self.tokenizer, expected_size), only_pages=True
        )
