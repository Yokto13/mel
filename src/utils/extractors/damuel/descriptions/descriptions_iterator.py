import json
from pathlib import Path
from typing import Callable

from utils.extractors.damuel.damuel_iterator import DamuelExtractor
from utils.extractors.damuel.parser import DamuelParser
from utils.extractors.data_type import DataType


class DamuelDescriptionsIterator(DamuelExtractor):
    # def __init__(
    #     self,
    # ):
    #     super().__init__(
    #         damuel_path, tokenizer, expected_size, is_name_ok
    #     )

    #     self.tokenizer = tokenizer
    #     assert self._check_token_in_tokenizer(name_token, tokenizer)

    #     self.name_token = name_token

    #     self.entry_processor = EntryProcessor(
    #         TokenizerWrapper(self.tokenizer, expected_size),
    #     )

    def _check_token_in_tokenizer(self, token, tokenizer):
        return token in tokenizer.get_vocab()

    def _iterate_file(self, f):
        for line in f:
            damuel_entry = json.loads(line)
            yield damuel_entry
