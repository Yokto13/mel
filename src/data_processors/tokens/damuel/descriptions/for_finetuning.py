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
        treat_qids_as_ints=True,
    ):
        super().__init__(
            damuel_path, tokenizer, expected_size, filename_is_ok, treat_qids_as_ints
        )

        self.tokenizer.add_special_tokens({"cls_token": name_token})
        self.name_token = name_token

        self.entry_processor = EntryProcessor(
            TokenizerWrapper(self.tokenizer, expected_size),
            self.qid_parser,
        )

    def _iterate_file(self, f):
        for line in f:
            damuel_entry = json.loads(line)
            result = self.entry_processor.process_to_one(damuel_entry, self.name_token)
            if result is not None:
                yield result
