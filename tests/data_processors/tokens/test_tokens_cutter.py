import logging
import pytest
from data_processors.tokens.tokenizer_wrapper import TokenizerWrapper
from data_processors.tokens.tokens_cutter import TokensCutter, fast_token_mention_span
from transformers import BertTokenizerFast


class TestTokensCutter:
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
        text = "This is a test text with a mention of [M]John Smith[M]."
        tokenizer_wrapper = TokenizerWrapper(tokenizer, expected_size=10)
        tokens_cutter = TokensCutter(
            text=text,
            tokenizer_wrapper=tokenizer_wrapper,
            expected_size=10,
            label_token=label_token,
        )
        result = tokens_cutter.cut_mention_with_context()

        assert len(result) == 10
        assert tokenizer.decode([tokenizer.vocab[label_token]]) == label_token
        assert list(result).count(tokenizer.vocab[label_token]) == 2

    def test_cut_mention_with_context_long_text(
        self, tokenizer: BertTokenizerFast, label_token: str
    ):
        long_text = (
            "This is a very long text with a mention of [M]John Smith[M]. "
            "It contains a lot of additional information that is not relevant to the mention. "
            "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed auctor, magna a bibendum bibendum, "
            "augue magna tincidunt augue, eget ultricies augue augue eget augue. Sed auctor, magna a bibendum bibendum, "
            "augue magna tincidunt augue, eget ultricies augue augue eget augue. Phasellus vestibulum lorem sed risus ultricies tristique. "
            "Nulla aliquet enim tortor at auctor urna nunc. Amet nisl suscipit adipiscing bibendum est ultricies integer quis auctor elit. "
            "Velit scelerisque in dictum non consectetur a erat nam. Pretium viverra suspendisse potenti nullam ac tortor vitae purus faucibus."
        )
        tokenizer_wrapper = TokenizerWrapper(tokenizer, expected_size=20)
        tokens_cutter = TokensCutter(
            text=long_text,
            tokenizer_wrapper=tokenizer_wrapper,
            expected_size=20,
            label_token=label_token,
        )
        result = tokens_cutter.cut_mention_with_context()

        assert len(result) == 20
        assert tokenizer.decode([tokenizer.vocab[label_token]]) == label_token
        assert list(result).count(tokenizer.vocab[label_token]) == 2

    def test_cut_mention_with_context_short_text(
        self, tokenizer: BertTokenizerFast, label_token: str
    ):
        short_text = "[M]John Smith[M] is mentioned."
        tokenizer_wrapper = TokenizerWrapper(tokenizer, expected_size=10)
        tokens_cutter = TokensCutter(
            text=short_text,
            tokenizer_wrapper=tokenizer_wrapper,
            expected_size=10,
            label_token=label_token,
        )
        result = tokens_cutter.cut_mention_with_context()

        assert len(result) == 10
        assert tokenizer.decode([tokenizer.vocab[label_token]]) == label_token
        assert list(result).count(tokenizer.vocab[label_token]) == 2

    def test_cut_mention_with_context_small_text_large_expected_size(
        self, tokenizer: BertTokenizerFast, label_token: str, caplog
    ):
        small_text = "[M]John Smith[M]"
        expected_size = 20

        with caplog.at_level(logging.WARNING):
            tokenizer_wrapper = TokenizerWrapper(tokenizer, expected_size=expected_size)
            tokens_cutter = TokensCutter(
                text=small_text,
                tokenizer_wrapper=tokenizer_wrapper,
                expected_size=expected_size,
                label_token=label_token,
            )
            result = tokens_cutter.cut_mention_with_context()

        assert len(result) == expected_size
        assert tokenizer.decode([tokenizer.vocab[label_token]]) == label_token
        assert list(result).count(tokenizer.vocab[label_token]) == 2

        assert len(caplog.text) > 0

    def test_cut_mention_with_context_large_mention(
        self, tokenizer: BertTokenizerFast, label_token: str
    ):
        text = "This is a text with a very large mention of [M]Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed auctor, magna a bibendum bibendum, augue magna tincidunt augue, eget ultricies augue augue eget augue. Sed auctor, magna a bibendum bibendum, augue magna tincidunt augue, eget ultricies augue augue eget augue.[M]"
        expected_size = 10
        tokenizer_wrapper = TokenizerWrapper(tokenizer, expected_size=expected_size)

        tokens_cutter = TokensCutter(
            text=text,
            tokenizer_wrapper=tokenizer_wrapper,
            expected_size=expected_size,
            label_token=label_token,
        )
        result = tokens_cutter.cut_mention_with_context()

        assert len(result) == expected_size
        assert tokenizer.decode([tokenizer.vocab[label_token]]) == label_token
        assert list(result).count(tokenizer.vocab[label_token]) == 2


class TestFastTokenMentionSpan:
    @pytest.fixture(scope="class")
    def label_token_id(self) -> int:
        return 3

    @pytest.mark.parametrize(
        "all_tokens, expected_output",
        [
            ([0, 1, 2, 3, 4, 3, 5, 6], slice(3, 6)),  # Mention found
            ([3, 3, 1, 2, 4, 5, 6], slice(0, 2)),  # Mention at the start
            ([0, 1, 2, 4, 5, 3, 3], slice(5, 7)),  # Mention at the end
        ],
    )
    def test_fast_token_mention_span(self, all_tokens, label_token_id, expected_output):
        assert fast_token_mention_span(all_tokens, label_token_id) == expected_output
