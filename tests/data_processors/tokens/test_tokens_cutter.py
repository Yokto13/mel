import logging
import pytest
from data_processors.tokens.tokenizer_wrapper import TokenizerWrapper
from data_processors.tokens.tokens_cutter import TokensCutterV3
from transformers import BertTokenizerFast


class TestTokensCutterV3:
    @pytest.fixture(scope="class")
    def tokenizer(self) -> BertTokenizerFast:
        tokenizer = BertTokenizerFast.from_pretrained("setu4993/LEALLA-base")
        tokenizer.add_tokens(["[M]"])
        return tokenizer

    @pytest.fixture(scope="class")
    def label_token(self) -> str:
        return "[M]"

    def test_cut_mention_with_context(
        self, tokenizer: BertTokenizerFast, label_token: str
    ):
        text = "This is a test text with a mention of John Smith."
        mention_slice = slice(29, 39)  # "John Smith"
        tokenizer_wrapper = TokenizerWrapper(tokenizer, expected_size=10)
        tokens_cutter = TokensCutterV3(
            text=text,
            tokenizer_wrapper=tokenizer_wrapper,
            expected_size=10,
            label_token=label_token,
        )
        result = tokens_cutter.cut_mention_with_context(mention_slice)

        assert len(result) == 10
        assert tokenizer.decode([tokenizer.vocab[label_token]]) == label_token
        assert result.count(tokenizer.vocab[label_token]) == 2

    def test_cut_mention_with_context_long_text(
        self, tokenizer: BertTokenizerFast, label_token: str
    ):
        long_text = (
            "This is a very long text with a mention of John Smith. "
            "It contains a lot of additional information that is not relevant to the mention."
        )
        mention_slice = slice(38, 48)
        tokenizer_wrapper = TokenizerWrapper(tokenizer, expected_size=20)
        tokens_cutter = TokensCutterV3(
            text=long_text,
            tokenizer_wrapper=tokenizer_wrapper,
            expected_size=20,
            label_token=label_token,
        )
        result = tokens_cutter.cut_mention_with_context(mention_slice)

        assert len(result) == 20
        assert tokenizer.decode([tokenizer.vocab[label_token]]) == label_token
        assert result.count(tokenizer.vocab[label_token]) == 2

    def test_cut_mention_with_context_short_text(
        self, tokenizer: BertTokenizerFast, label_token: str
    ):
        short_text = "John Smith is mentioned."
        mention_slice = slice(0, 10)  # "John Smith"
        tokenizer_wrapper = TokenizerWrapper(tokenizer, expected_size=10)
        tokens_cutter = TokensCutterV3(
            text=short_text,
            tokenizer_wrapper=tokenizer_wrapper,
            expected_size=10,
            label_token=label_token,
        )
        result = tokens_cutter.cut_mention_with_context(mention_slice)

        assert len(result) == 10
        assert tokenizer.decode([tokenizer.vocab[label_token]]) == label_token
        assert result.count(tokenizer.vocab[label_token]) == 2

    def test_cut_mention_with_context_small_text_large_expected_size(
        self, tokenizer: BertTokenizerFast, label_token: str, caplog
    ):
        small_text = "John Smith"
        mention_slice = slice(0, 10)  # "John Smith"
        expected_size = 20

        with caplog.at_level(logging.WARNING):
            tokenizer_wrapper = TokenizerWrapper(tokenizer, expected_size=expected_size)
            tokens_cutter = TokensCutterV3(
                text=small_text,
                tokenizer_wrapper=tokenizer_wrapper,
                expected_size=expected_size,
                label_token=label_token,
            )
            result = tokens_cutter.cut_mention_with_context(mention_slice)

        assert len(result) < expected_size
        assert tokenizer.decode([tokenizer.vocab[label_token]]) == label_token
        assert result.count(tokenizer.vocab[label_token]) == 2

        assert len(caplog.text) > 0

    def test_cut_mention_with_context_large_mention(
        self, tokenizer: BertTokenizerFast, label_token: str
    ):
        text = "This is a text with a very large mention of Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed auctor, magna a bibendum bibendum, augue magna tincidunt augue, eget ultricies augue augue eget augue. Sed auctor, magna a bibendum bibendum, augue magna tincidunt augue, eget ultricies augue augue eget augue."
        mention_slice = slice(32, 253)  # The large mention
        expected_size = 10
        tokenizer_wrapper = TokenizerWrapper(tokenizer, expected_size=expected_size)

        tokens_cutter = TokensCutterV3(
            text=text,
            tokenizer_wrapper=tokenizer_wrapper,
            expected_size=expected_size,
            label_token=label_token,
        )
        result = tokens_cutter.cut_mention_with_context(mention_slice)

        assert len(result) == expected_size
        assert tokenizer.decode([tokenizer.vocab[label_token]]) == label_token
        assert result.count(tokenizer.vocab[label_token]) == 2
