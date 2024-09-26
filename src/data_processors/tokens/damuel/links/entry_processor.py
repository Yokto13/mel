from src.data_processors.tokens.tokenizer_wrapper import TokenizerWrapper
from src.data_processors.tokens.tokens_cutter import TokensCutter


class EntryProcessor:
    def __init__(self, tokenizer_wrapper, only_wiki):
        self.tokenizer_wrapper = tokenizer_wrapper
        self.only_wiki = only_wiki

    def process_both(self, damuel_entry: dict) -> list[tuple]:
        if "wiki" not in damuel_entry:
            return None
        wiki = damuel_entry["wiki"]
        wiki_processor = WikiProcessorBoth(
            self.tokenizer_wrapper,
            wiki,
            self.only_wiki,
        )
        return list(wiki_processor)

    def process_to_one(self, damuel_entry: dict, mention_token: str) -> list:
        if "wiki" not in damuel_entry:
            return None
        wiki = damuel_entry["wiki"]
        wiki_processor = WikiProcessorFinetuning(
            self.tokenizer_wrapper, wiki, self.only_wiki, mention_token
        )
        return list(wiki_processor)


class WikiProcessorBoth:
    def __init__(
        self,
        tokenizer_wrapper: TokenizerWrapper,
        wiki,
        only_wiki: bool,
        init_token_cutter=True,
    ) -> None:
        self.wiki = wiki
        self.text = wiki["text"]
        self.tokenizer_wrapper = tokenizer_wrapper
        self.tokens_cutter = None
        if init_token_cutter:
            self.tokens_cutter = TokensCutter(
                self.text, tokenizer_wrapper, tokenizer_wrapper.expected_size
            )
        self.only_wiki = only_wiki
        self.damuel_tokens = wiki["tokens"]

    def __iter__(self):
        for link in self.wiki["links"]:
            if self._should_skip_link(link):
                continue
            link_processed = self.process_link(link)
            if link_processed is not None:
                yield link_processed

    def process_link(self, link):
        qid = int(link["qid"][1:])

        start = link["start"]
        end = link["end"] - 1
        try:
            mention_slice_chars = slice(
                self.damuel_tokens[start]["start"], self.damuel_tokens[end]["end"]
            )
        except IndexError:
            print("Index Error, skipping")
            return None

        try:
            cutted_tokens = self.tokens_cutter.cut_mention_with_context(
                mention_slice_chars
            )
        except TypeError as e:
            print(e)
            print(f"Error in mention: {self.text[mention_slice_chars]}")
            return None
        mention_pair = (cutted_tokens, qid)

        entity_name_tokens = self.tokens_cutter.cut_mention_name(mention_slice_chars)
        entity_name_pair = (entity_name_tokens, qid)

        return entity_name_pair, mention_pair

    def _should_skip_link(self, link):
        if "qid" not in link:
            return True
        if self.only_wiki and link["origin"] != "wiki":
            return True
        return False


class WikiProcessorFinetuning(WikiProcessorBoth):
    def __init__(
        self,
        tokenizer_wrapper: TokenizerWrapper,
        wiki,
        only_wiki: bool,
        mention_token: str,
        expected_chars_per_token: int = 5,
    ) -> None:
        super().__init__(tokenizer_wrapper, wiki, only_wiki, False)

        self.expected_chars_per_token = expected_chars_per_token
        self.char_window = (
            expected_chars_per_token * tokenizer_wrapper.expected_size * 2
        )
        self.mention_token = mention_token

    def process_link(self, link):
        qid = int(link["qid"][1:])

        start = link["start"]
        end = link["end"] - 1

        try:
            mention_slice_chars = slice(
                self.damuel_tokens[start]["start"], self.damuel_tokens[end]["end"]
            )
        except IndexError:
            print("Index Error, skipping...")
            return None

        smaller_text, mention_slice_chars = self._apply_char_window(
            self.text, mention_slice_chars
        )

        text_with_special_around_mention, mention_slice_chars = (
            self._add_token_around_mention(
                smaller_text, mention_slice_chars, self.mention_token
            )
        )

        # Ugly side stepping of tokenizer wrapper
        self.token_cutter = TokensCutter(
            text_with_special_around_mention,
            self.tokenizer_wrapper,
            self.tokenizer_wrapper.expected_size,
        )

        cutted_tokens = self.token_cutter.cut_mention_with_context(mention_slice_chars)
        return (cutted_tokens, qid)

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
