import logging
from functools import partial

import numba as nb
import numpy as np

_logger = logging.getLogger(__name__)


def fast_token_mention_span(all_tokens, label_token_id):
    # @nb.njit
    def _fast_token_mention_span(all_tokens, label_token_id):
        mention_start_idx, mention_end_idx = None, None
        for i, token in enumerate(all_tokens):
            if token == label_token_id:
                if mention_start_idx is None:
                    mention_start_idx = i
                else:
                    mention_end_idx = i + 1
                    break
        assert (
            mention_start_idx is not None and mention_end_idx is not None
        ), f"Mention not found: {label_token_id}, {all_tokens}"
        return mention_start_idx, mention_end_idx

    pre_slice = _fast_token_mention_span(all_tokens, label_token_id)
    return slice(pre_slice[0], pre_slice[1])


class TokensCutter:
    def __init__(
        self, text, tokenizer_wrapper, expected_size, label_token, padding_token_id=0
    ):
        self.text = text
        # self.tokenizer = tokenizer
        self.tokenizer_wrapper = tokenizer_wrapper
        self.expected_size = expected_size

        self.padding_token_id = padding_token_id

        # Sometimes sidestepping the wrapper is necessery unless we want to rewrite old code.
        self.be_of_all = self.tokenizer_wrapper.tokenizer(
            text,
            return_tensors="np",
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
        if self._contains_padding(self.be_of_all["input_ids"][0]):
            self._warn_about_padding()
        self.all_tokens = self.be_of_all["input_ids"][0]
        self.offset_mapping = self.be_of_all["offset_mapping"][0]
        self.label_token_id = self.tokenizer_wrapper.tokenizer.encode(
            label_token, add_special_tokens=False
        )[0]

    def cut_mention_with_context(self):
        entity_name_slice_in_tokens = fast_token_mention_span(
            self.all_tokens, self.label_token_id
        )
        if self._is_entity_name_too_large(entity_name_slice_in_tokens):
            entity_name_slice_in_tokens = slice(
                entity_name_slice_in_tokens.start,
                min(
                    entity_name_slice_in_tokens.stop,
                    entity_name_slice_in_tokens.start + self.size_no_special_tokens,
                ),
            )
        return self._cut(entity_name_slice_in_tokens)

    @property
    def size_no_special_tokens(self):
        return self.expected_size - 2

    def _contains_padding(self, tokens):
        return np.sum(tokens == self.padding_token_id) > 0

    def _is_entity_name_too_large(
        self, entity_name_slice_in_tokens, max_entity_name_tokens
    ):
        return (
            entity_name_slice_in_tokens.stop - entity_name_slice_in_tokens.start
            > max_entity_name_tokens
        )

    def _cut(self, entity_name_slice_in_tokens):
        cut_f = self._choose_cut_method(entity_name_slice_in_tokens)
        return cut_f()

    def _count_remaining_for_context(self, entity_name_slice_in_tokens):
        return self.size_no_special_tokens - (
            entity_name_slice_in_tokens.stop - entity_name_slice_in_tokens.start
        )

    def _choose_cut_method(self, entity_name_slice_in_tokens):
        remains_for_context = self._count_remaining_for_context(
            entity_name_slice_in_tokens
        )

        left_context_start = (
            entity_name_slice_in_tokens.start - remains_for_context // 2
        )
        right_context_end = entity_name_slice_in_tokens.stop + (
            remains_for_context - remains_for_context // 2
        )

        can_cut_from_middle = (
            left_context_start >= 0
            and right_context_end <= len(self.all_tokens)
            or remains_for_context <= 0
        )

        if can_cut_from_middle:
            return partial(self._mid_cut, left_context_start, right_context_end)
        elif left_context_start < 0:
            return self._more_on_right_cut
        else:
            return self._more_on_left_cut

    def _mid_cut(self, left, right):
        char_start = self.be_of_all.token_to_chars(left).start
        char_end = self.be_of_all.token_to_chars(right - 1).end
        return self.tokenizer_wrapper.tokenize(
            self.text[char_start:char_end],
            max_length=self.expected_size,
        )

    def _more_on_right_cut(self):
        end_tok_candidate = min(
            self.size_no_special_tokens - 1, len(self.all_tokens) - 1
        )
        char_end = self.be_of_all.token_to_chars(end_tok_candidate)
        return self.tokenizer_wrapper.tokenize(
            self.text[: char_end.end],
            max_length=self.expected_size,
        )

    def _more_on_left_cut(self):
        start_tok_candidate = max(0, len(self.all_tokens) - self.size_no_special_tokens)
        char_start = self.be_of_all.token_to_chars(start_tok_candidate)
        return self.tokenizer_wrapper.tokenize(
            self.text[char_start.start :],
            max_length=self.expected_size,
        )

    def _warn_about_padding(self):
        _logger.warning(
            "Padding tokens are present in the input text. This means that input text is shorter than expected."
        )
