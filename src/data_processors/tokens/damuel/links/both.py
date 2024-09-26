import json
from pathlib import Path
from typing import Callable

from data_processors.tokens.damuel.damuel_iterator import DamuelLinksIterator
from data_processors.tokens.tokenizer_wrapper import TokenizerWrapper
from src.data_processors.tokens.damuel.links.entry_processor import EntryProcessor


class DamuelLinksTokensIteratorBoth(DamuelLinksIterator):
    def __init__(
        self,
        damuel_path,
        tokenizer,
        only_wiki=True,
        expected_size=64,
        filename_is_ok: Callable[[str], bool] = None,
    ):
        super().__init__(
            damuel_path,
            tokenizer,
            only_wiki,
            expected_size,
            filename_is_ok,
        )

        self.entry_processor = EntryProcessor(
            TokenizerWrapper(tokenizer, expected_size), only_wiki
        )

    def _iterate_file(self, f):
        for line in f:
            damuel_entry = json.loads(line)
            result = self.entry_processor.process_both(damuel_entry)
            if result is not None:
                for link in result:
                    yield link
