from functools import partial
from itertools import zip_longest

import numba as nb
import numpy as np


class TokensCutter:
    def __init__(self, text, tokenizer_wrapper, expected_size):
        self.text = text
        # self.tokenizer = tokenizer
        self.tokenizer_wrapper = tokenizer_wrapper
        self.expected_size = expected_size

        # Sometimes sidestepping the wrapper is necessery unless we want to rewrite old code.
        self.be_of_all = self.tokenizer_wrapper.tokenizer(
            text,
            return_tensors="np",
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
        self.all_tokens = self.be_of_all["input_ids"][0]
        self.offset_mapping = self.be_of_all["offset_mapping"][0]

    def set_text(self, text):
        self.text = text
        self.be_of_all = self.tokenizer_wrapper.tokenizer(
            text, return_tensors="np", add_special_tokens=False
        )
        self.all_tokens = self.be_of_all["input_ids"][0]

    def cut_mention_name(self, entity_name_slice_in_chars):
        mention_text = self.text[entity_name_slice_in_chars]
        return self.tokenizer_wrapper.tokenize(
            mention_text,
            max_length=self.expected_size,
        )

    def cut_mention_with_context(self, entity_name_slice_in_chars):
        while True:
            try:
                entity_name_slice_in_tokens = self._get_token_mention_span(
                    entity_name_slice_in_chars
                )
                # entity_name_slice_in_tokens = (
                # self._get_token_mention_span_with_offset_mapping(
                # entity_name_slice_in_chars
                # )
                # )
            except TypeError:
                print("Hacking entity name slice in chars...")
                print(entity_name_slice_in_chars)
                print(self.text)
                entity_name_slice_in_chars = slice(
                    entity_name_slice_in_chars.start,
                    entity_name_slice_in_chars.stop - 1,
                )
            else:
                break

        return self._cut(entity_name_slice_in_tokens)

    @property
    def size_no_special_tokens(self):
        return self.expected_size - 2

    def _get_token_mention_span(self, entity_name_slice_in_chars):
        mention_start_idx = self.be_of_all.char_to_token(
            entity_name_slice_in_chars.start
        )
        mention_end_idx = (
            self.be_of_all.char_to_token(entity_name_slice_in_chars.stop - 1) + 1
        )
        return slice(mention_start_idx, mention_end_idx)

    def _get_token_mention_span_with_offset_mapping(self, entity_name_slice_in_chars):
        # print(entity_name_slice_in_chars)
        # print(self.offset_mapping)
        mention_start_idx = None
        mention_end_idx = None
        for i, (start, end) in enumerate(self.offset_mapping):
            if start <= entity_name_slice_in_chars.start < end:
                mention_start_idx = i
            if start <= entity_name_slice_in_chars.stop - 1 < end:
                mention_end_idx = i + 1
        if mention_start_idx is None or mention_end_idx is None:
            print(entity_name_slice_in_chars)
            print(self.offset_mapping)
            print(len(self.text))
            print(mention_start_idx, mention_end_idx)
            print(self.offset_mapping[mention_start_idx : mention_start_idx + 20])
            raise ValueError(entity_name_slice_in_chars, self.text)
        return slice(mention_start_idx, mention_end_idx)

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


def iterate_by_two(iterable):
    it = iter(iterable)
    return zip(it, it)


def iterate_indices(l_start, r_start, max_length):
    return zip_longest(
        range(min(l_start, max_length - 1), -1, -1),
        range(r_start, max_length),
        fillvalue=None,
    )


class TokensCutterV2:
    def __init__(self, text, tokenizer, expected_size, mention_token):
        self.text = text
        self.tokenizer = tokenizer
        self.expected_size = expected_size
        self.size_without_special_tokens = expected_size - 2
        self.mention_token_id = self.tokenizer.encode(
            mention_token, add_special_tokens=False
        )[0]
        self.tokens = self.tokenizer.encode(text, return_tensors="np")[0]

        self.start_token = self.tokenizer.encode(text, add_special_tokens=True)[0]
        self.end_token = self.tokenizer.encode(text, add_special_tokens=True)[-1]

        self.mention_idx_to_slice = self._get_mention_idx_to_slice(
            self.tokens, self.mention_token_id
        )

    def cut(self, mention_idx: int):
        return_mask = self._get_return_mask(mention_idx)

        res_tokens = np.empty(self.expected_size, dtype=np.int32)
        res_tokens[0] = self.start_token
        res_tokens[-1] = self.end_token

        res_tokens[1 : 1 + sum(return_mask)] = self.tokens[return_mask]

        return res_tokens

    def _get_return_mask(self, mention_idx):
        return_mask = np.full(len(self.tokens), dtype=np.bool_, fill_value=False)
        mention_slice = self.mention_idx_to_slice[mention_idx]

        if (
            mention_slice.stop - mention_slice.start
        ) > self.size_without_special_tokens:  # problem mention is too large should not happen often
            start_idx = mention_slice.start
            end_idx = start_idx + self.size_without_special_tokens - 1
            return_mask[start_idx:end_idx] = True
            return_mask[mention_slice.stop - 1] = (
                True  # this adds the right mention token to the return mask
            )
            return return_mask

        return_mask[mention_slice] = True

        remains_to_fill = (
            self.size_without_special_tokens - mention_slice.stop + mention_slice.start
        )

        if remains_to_fill > 0:
            for start_idx, end_idx in iterate_indices(
                mention_slice.start - 1,
                mention_slice.stop,
                len(self.tokens),
            ):
                if start_idx is None and end_idx is None:
                    break
                if (
                    start_idx is not None
                    and self.tokens[start_idx] != self.mention_token_id
                ):
                    return_mask[start_idx] = True
                    remains_to_fill -= 1
                if remains_to_fill == 0:
                    break
                if (
                    end_idx is not None
                    and self.tokens[end_idx] != self.mention_token_id
                ):
                    return_mask[end_idx] = True
                    remains_to_fill -= 1
                if remains_to_fill == 0:
                    break
        return return_mask

    def _get_mention_idx_to_slice(self, tokens, mention_token_id):
        mention_idxs = np.where(tokens == mention_token_id)[0]
        assert len(mention_idxs) % 2 == 0, "Mention token idxs should be even"
        mention_idxs_to_slice = [
            slice(idx1, idx2 + 1) for idx1, idx2 in iterate_by_two(mention_idxs)
        ]
        return mention_idxs_to_slice


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


class TokensCutterV3:
    def __init__(self, text, tokenizer_wrapper, expected_size, label_token):
        self.text = text
        # self.tokenizer = tokenizer
        self.tokenizer_wrapper = tokenizer_wrapper
        self.expected_size = expected_size

        # Sometimes sidestepping the wrapper is necessery unless we want to rewrite old code.
        self.be_of_all = self.tokenizer_wrapper.tokenizer(
            text,
            return_tensors="np",
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
        self.all_tokens = self.be_of_all["input_ids"][0]
        self.offset_mapping = self.be_of_all["offset_mapping"][0]
        self.label_token_id = self.tokenizer_wrapper.tokenizer.encode(
            label_token, add_special_tokens=False
        )[0]

    def cut_mention_with_context(self, entity_name_slice_in_chars):
        entity_name_slice_in_tokens = fast_token_mention_span(
            self.all_tokens, self.label_token_id
        )
        while True:
            try:
                entity_name_slice_in_tokens = fast_token_mention_span(
                    self.all_tokens, self.label_token_id
                )
                # entity_name_slice_in_tokens = (
                # self._get_token_mention_span_with_offset_mapping(
                # entity_name_slice_in_chars
                # )
                # )
            except TypeError:
                print("Hacking entity name slice in chars...")
                print(entity_name_slice_in_chars)
                print(self.text)
                entity_name_slice_in_chars = slice(
                    entity_name_slice_in_chars.start,
                    entity_name_slice_in_chars.stop - 1,
                )
            else:
                break

        return self._cut(entity_name_slice_in_tokens)

    @property
    def size_no_special_tokens(self):
        return self.expected_size - 2

    def _get_token_mention_span(self, entity_name_slice_in_chars):
        mention_start_idx, mention_end_idx = None, None
        for i, token in enumerate(self.all_tokens):
            if token == self.label_token_id:
                if mention_start_idx is None:
                    mention_start_idx = i
                else:
                    mention_end_idx = i + 1
                    break
        assert (
            mention_start_idx is not None and mention_end_idx is not None
        ), f"Mention not found: {self.label_token_id}, {self.all_tokens}"
        return slice(mention_start_idx, mention_end_idx)

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
