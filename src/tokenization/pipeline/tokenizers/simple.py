from collections.abc import Generator

from .base import TokenizerStep


class SimpleTokenizer(TokenizerStep):
    def __init__(self, tokenizer, expected_size):
        super().__init__(tokenizer, expected_size)

    def process(
        self, input_gen: Generator[tuple, None, None]
    ) -> Generator[tuple, None, None]:
        for mention, qid in input_gen:
            tokens = self.tokenizer_wrapper.tokenize(mention)
            yield tokens, qid
