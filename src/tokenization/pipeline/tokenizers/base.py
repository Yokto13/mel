from abc import ABC

from data_processors.tokens.tokenizer_wrapper import TokenizerWrapper

from ..base import PipelineStep


class TokenizerStep(PipelineStep, ABC):
    def __init__(self, tokenizer, expected_size: int):
        self.tokenizer = tokenizer
        self.expected_size = expected_size
        self.tokenizer_wrapper = TokenizerWrapper(tokenizer, expected_size)
