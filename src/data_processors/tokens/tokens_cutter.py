from functools import partial


class TokensCutter:
    def __init__(self, text, tokenizer_wrapper, expected_size):
        self.text = text
        # self.tokenizer = tokenizer
        self.tokenizer_wrapper = tokenizer_wrapper
        self.expected_size = expected_size

        # Sometimes sidestepping the wrapper is necessery unless we want to rewrite old code.
        self.be_of_all = self.tokenizer_wrapper.tokenizer(
            text, return_tensors="np", add_special_tokens=False
        )
        self.all_tokens = self.be_of_all["input_ids"][0]

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
        try:
            entity_name_slice_in_tokens = self._get_token_mention_span(
                entity_name_slice_in_chars
            )
        except TypeError:
            raise TypeError(entity_name_slice_in_chars, self.text)

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
