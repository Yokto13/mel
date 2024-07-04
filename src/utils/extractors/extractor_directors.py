from dataclasses import dataclass
from typing import Optional
from transformers import PretrainedTokenizerBase

from utils.extractors.extractor_builder import DamuelDescriptionsExtractorBuilder
from utils.extractors.tokenizer_wrapper import TokenizerWrapper
from utils.extractors.damuel.descriptions.entry_processor import DescriptionTokenizer


@dataclass
class TokenizingParams:
    tokenizer: PretrainedTokenizerBase
    size: int
    mention_token: Optional[str]


class Director:
    def construct_descriptions_for_finetuning(
        self,
        damuel_path: str,
        tokenizing_params,
        n_of_extractors=None,
        current_extractor_n=None,
    ):
        builder = DamuelDescriptionsExtractorBuilder()
        builder.set_source(damuel_path)
        if n_of_extractors is not None and current_extractor_n is not None:
            assert 0 <= current_extractor_n < n_of_extractors
            builder.set_modulo_file_acceptor(n_of_extractors, current_extractor_n)
        builder.set_entry_processor(
            self._get_description_entry_processor(tokenizing_params)
        )

    @classmethod
    def _get_tokenizer_wrapper(cls, tokenizing_params: TokenizingParams):
        assert cls._check_token_in_tokenizer(
            tokenizing_params.tokenizer, tokenizing_params.mention_token
        )
        return TokenizerWrapper(tokenizing_params.tokenizer, tokenizing_params.size)

    @classmethod
    def _get_description_entry_processor(cls, tokenizing_params: TokenizingParams):
        tokenizer_wrapper = cls._get_tokenizer_wrapper(tokenizing_params)
        return DescriptionTokenizer(tokenizer_wrapper)

    @classmethod
    def _check_token_in_tokenizer(
        cls, tokenizer: PretrainedTokenizerBase, token: str
    ) -> bool:
        return token in tokenizer.get_vocab()
