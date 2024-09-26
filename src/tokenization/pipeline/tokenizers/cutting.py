from collections.abc import Generator
from data_processors.tokens.tokens_cutter import TokensCutterV3

from .base import TokenizerStep


class CuttingTokenizer(TokenizerStep):
    def __init__(
        self, tokenizer, expected_size, label_token, expected_chars_per_token=11
    ):
        super().__init__(tokenizer, expected_size)
        self.label_token = label_token
        self.expected_chars_per_token = expected_chars_per_token
        self.char_window = expected_chars_per_token * expected_size

    def process(
        self, input_gen: Generator[tuple, None, None]
    ) -> Generator[tuple, None, None]:
        for mention_slice, text, qid in input_gen:
            text, mention_slice_chars = self._apply_char_window(text, mention_slice)
            text, mention_slice_chars = self._add_token_around_mention(
                text, mention_slice_chars, self.label_token
            )
            tokens_cutter = TokensCutterV3(
                text, self.tokenizer_wrapper, self.expected_size, self.label_token
            )
            yield tokens_cutter.cut_mention_with_context(mention_slice_chars), qid

    def _apply_char_window(self, text, mention_slice_chars):
        """Cuts char_window chars around mention from the text and recalculates the slice.

        Returns:
            new_text: str
            new_slice_chars: slice
        """
        start = max(0, mention_slice_chars.start - self.char_window // 2)
        end = min(len(text), mention_slice_chars.stop + self.char_window // 2)
        new_text = text[start:end]
        new_slice_chars = slice(
            mention_slice_chars.start - start, mention_slice_chars.stop - start
        )
        assert text[mention_slice_chars] == new_text[new_slice_chars]
        return new_text, new_slice_chars

    def _add_token_around_mention(self, text, mention_slice, token):
        new_text = f"{text[:mention_slice.start]}{token} {text[mention_slice]} {token}{text[mention_slice.stop:]}"
        new_slice = slice(mention_slice.start, mention_slice.stop + 2 * len(token) + 2)
        return new_text, new_slice
