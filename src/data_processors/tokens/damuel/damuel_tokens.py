from collections import defaultdict
import json
import os
from pathlib import Path
from typing import Callable
import lzma

from data_processors.tokens.mention_qid_pair import MentionQidPair
from data_processors.tokens.tokens_cutter import TokensCutter


def _add_token_around_mention(text, mention_slice, token):
    new_text = f"{text[:mention_slice.start]}{token} {text[mention_slice]} {token}{text[mention_slice.stop:]}"
    new_slice = slice(mention_slice.start, mention_slice.stop + 2 * len(token) + 2)
    return new_text, new_slice


class LinksTokensIterator:
    def __init__(
        self,
        damuel_path,
        tokenizer,
        use_context=True,
        expected_size=64,
        filename_is_ok: Callable[[str], bool] = None,
        only_wiki=True,
        treat_qids_as_ints=True,
    ):
        self.damuel_path = damuel_path
        if not isinstance(damuel_path, Path):
            self.damuel_path = Path(damuel_path)
        self.use_context = use_context
        if self.use_context:
            assert (
                expected_size > 0
            ), "max_context_size must be greater than 0 when use_context is True"
        self.expected_size = expected_size
        self.tokenizer = tokenizer
        self.filename_is_ok = filename_is_ok
        self.only_wiki = only_wiki
        self.treat_qids_as_ints = treat_qids_as_ints
        self.qid_parser = self._get_qid_parser()

    def __iter__(self):
        for file in sorted(list(self.damuel_path.iterdir())):
            if (
                file.is_file()
                and file.name.startswith("part")
                and (self.filename_is_ok is None or self.filename_is_ok(file.name))
            ):
                print(file)
                if file.name.endswith(".xz"):
                    with lzma.open(file, "rt") as f:
                        for mention in self._iterate_file(f):
                            yield mention
                else:
                    with file.open("r") as f:
                        for mention in self._iterate_file(f):
                            yield mention

    def _qid_to_int(self, qid):
        return int(qid[1:])

    def _qid_to_str(self, qid):
        return qid

    def _get_qid_parser(self):
        if self.treat_qids_as_ints:
            return self._qid_to_int
        else:
            return self._qid_to_str

    def _iterate_file(self, f):
        for line in f:  # iter(file_map.readline, b""):
            # damuel_entry = json.loads(line.decode("utf-8"))
            damuel_entry = json.loads(line)
            if "wiki" not in damuel_entry and self.only_wiki:
                continue
            wiki = damuel_entry["wiki"]

            for mention_context in self._iterate_wiki(wiki):
                yield mention_context

    def _iterate_wiki(self, wiki):
        damuel_tokens = wiki["tokens"]
        text = wiki["text"]

        tokens_cutter = TokensCutter(text, self.tokenizer, self.expected_size)

        for link in wiki["links"]:
            if "qid" not in link:
                continue
            if self.only_wiki and link["origin"] != "wiki":
                continue

            qid = self.qid_parser(link["qid"])

            start = link["start"]
            end = link["end"] - 1
            mention_slice_chars = slice(
                damuel_tokens[start]["start"], damuel_tokens[end]["end"]
            )

            if self.use_context:
                try:
                    cutted_tokens = tokens_cutter.cut_mention_with_context(
                        mention_slice_chars
                    )
                except TypeError:
                    print(f"Error in mention: {text[mention_slice_chars]}")
                    continue
                yield MentionQidPair(cutted_tokens, qid)
            else:
                entity_name_tokens = tokens_cutter.cut_mention_name(mention_slice_chars)
                yield MentionQidPair(entity_name_tokens, qid)


class LinksTokensIteratorBoth:
    def __init__(
        self,
        damuel_path,
        tokenizer,
        expected_size=64,
        filename_is_ok: Callable[[str], bool] = None,
        only_wiki=True,
        treat_qids_as_ints=True,
        max_per_qid=float("inf"),
    ):
        self.damuel_path = damuel_path
        if not isinstance(damuel_path, Path):
            self.damuel_path = Path(damuel_path)
        self.expected_size = expected_size
        self.tokenizer = tokenizer
        self.filename_is_ok = filename_is_ok
        self.only_wiki = only_wiki
        self.treat_qids_as_ints = treat_qids_as_ints
        self.qid_parser = self._get_qid_parser()

        self.max_per_qid = max_per_qid
        self.qid_occurrences = defaultdict(int)

    def __iter__(self):
        for file in sorted(list(self.damuel_path.iterdir())):
            if (
                file.is_file()
                and file.name.startswith("part")
                and (self.filename_is_ok is None or self.filename_is_ok(file.name))
            ):
                print(file)
                if file.name.endswith(".xz"):
                    with lzma.open(file, "rt") as f:
                        for mention in self._iterate_file(f):
                            yield mention
                else:
                    with file.open("r") as f:
                        for mention in self._iterate_file(f):
                            yield mention

    def _qid_to_int(self, qid):
        return int(qid[1:])

    def _qid_to_str(self, qid):
        return qid

    def _get_qid_parser(self):
        if self.treat_qids_as_ints:
            return self._qid_to_int
        else:
            return self._qid_to_str

    def _iterate_file(self, f):
        for line in f:  # iter(file_map.readline, b""):
            # damuel_entry = json.loads(line.decode("utf-8"))
            damuel_entry = json.loads(line)
            if "wiki" not in damuel_entry and self.only_wiki:
                continue
            wiki = damuel_entry["wiki"]

            for mention_context in self._iterate_wiki(wiki):
                yield mention_context

    def _iterate_wiki(self, wiki):
        damuel_tokens = wiki["tokens"]
        text = wiki["text"]

        tokens_cutter = TokensCutter(text, self.tokenizer, self.expected_size)

        for link in wiki["links"]:
            if "qid" not in link:
                continue
            if self.only_wiki and link["origin"] != "wiki":
                continue

            qid = self.qid_parser(link["qid"])

            self.qid_occurrences[qid] += 1
            assert self.max_per_qid > 10**10
            if self.qid_occurrences[qid] > self.max_per_qid:
                continue

            start = link["start"]
            end = link["end"] - 1
            mention_slice_chars = slice(
                damuel_tokens[start]["start"], damuel_tokens[end]["end"]
            )

            try:
                cutted_tokens = tokens_cutter.cut_mention_with_context(
                    mention_slice_chars
                )
            except TypeError:
                print(f"Error in mention: {text[mention_slice_chars]}")
                continue
            mention_pair = MentionQidPair(cutted_tokens, qid)

            entity_name_tokens = tokens_cutter.cut_mention_name(mention_slice_chars)
            entity_name_pair = MentionQidPair(entity_name_tokens, qid)

            yield entity_name_pair, mention_pair


class DamuelTokensIteratorLimitByQid(LinksTokensIterator):
    def __init__(self, *args, max_per_qid, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_per_qid = max_per_qid
        self.qid_occurrences = defaultdict(int)

    def __iter__(self):
        for x in super().__iter__():
            if self.qid_occurrences[x.qid] < self.max_per_qid:
                self.qid_occurrences[x.qid] += 1
                yield x


class LinksTokensIteratorTogether:
    def __init__(
        self,
        damuel_path,
        tokenizer,
        expected_size=64,
        filename_is_ok: Callable[[str], bool] = None,
        only_wiki=True,
        treat_qids_as_ints=True,
        mention_token="[M]",
        max_per_qid=float("inf"),
        expected_chars_per_token=5,
    ):
        self.damuel_path = damuel_path
        if not isinstance(damuel_path, Path):
            self.damuel_path = Path(damuel_path)
        self.expected_size = expected_size

        self.tokenizer = tokenizer
        self.tokenizer.add_special_tokens({"cls_token": mention_token})
        self.mention_token = mention_token

        self.filename_is_ok = filename_is_ok
        self.only_wiki = only_wiki
        self.treat_qids_as_ints = treat_qids_as_ints
        self.qid_parser = self._get_qid_parser()

        self.max_per_qid = max_per_qid
        self.qid_occurrences = defaultdict(int)
        self.expected_chars_per_token = expected_chars_per_token
        self.char_window = (
            expected_chars_per_token * expected_size * 2
        )  # Multiply by 2 to get some safety buffer

    def __iter__(self):
        for file in sorted(list(self.damuel_path.iterdir())):
            if (
                file.is_file()
                and file.name.startswith("part")
                and (self.filename_is_ok is None or self.filename_is_ok(file.name))
            ):
                if file.name.endswith(".xz"):
                    with lzma.open(file, "rt") as f:
                        for mention in self._iterate_file(f):
                            yield mention
                else:
                    with file.open("r") as f:
                        for mention in self._iterate_file(f):
                            yield mention

    def _qid_to_int(self, qid):
        return int(qid[1:])

    def _qid_to_str(self, qid):
        return qid

    def _get_qid_parser(self):
        if self.treat_qids_as_ints:
            return self._qid_to_int
        else:
            return self._qid_to_str

    def _iterate_file(self, f):
        for line in f:  # iter(file_map.readline, b""):
            # damuel_entry = json.loads(line.decode("utf-8"))
            damuel_entry = json.loads(line)
            if "wiki" not in damuel_entry and self.only_wiki:
                continue
            wiki = damuel_entry["wiki"]

            for mention_context in self._iterate_wiki(wiki):
                yield mention_context

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

    def _iterate_wiki(self, wiki):
        damuel_tokens = wiki["tokens"]
        text = wiki["text"]

        for link in wiki["links"]:
            if "qid" not in link:
                continue
            if self.only_wiki and link["origin"] != "wiki":
                continue

            qid = self.qid_parser(link["qid"])

            start = link["start"]
            end = link["end"] - 1
            mention_slice_chars = slice(
                damuel_tokens[start]["start"], damuel_tokens[end]["end"]
            )

            # print(f"ORIGINAL:{text[mention_slice_chars]}")

            smaller_text, mention_slice_chars = self._apply_char_window(
                text, mention_slice_chars
            )

            text_with_special_around_mention, mention_slice_chars = (
                _add_token_around_mention(
                    smaller_text, mention_slice_chars, self.mention_token
                )
            )

            token_cutter = TokensCutter(
                text_with_special_around_mention, self.tokenizer, self.expected_size
            )

            cutted_tokens = token_cutter.cut_mention_with_context(mention_slice_chars)
            yield MentionQidPair(cutted_tokens, qid)


class DescriptionTokensIteratorTogether:
    def __init__(
        self,
        damuel_path,
        tokenizer,
        expected_size=64,
        filename_is_ok: Callable[[str], bool] = None,
        only_wiki=True,
        treat_qids_as_ints=True,
        class_token="[M]",
        max_per_qid=float("inf"),
        **kwargs,
    ):
        self.damuel_path = damuel_path
        if not isinstance(damuel_path, Path):
            self.damuel_path = Path(damuel_path)
        self.expected_size = expected_size

        self.tokenizer = tokenizer
        self.tokenizer.add_special_tokens({"cls_token": class_token})
        self.class_token = class_token

        self.filename_is_ok = filename_is_ok
        self.only_wiki = only_wiki
        self.treat_qids_as_ints = treat_qids_as_ints
        self.qid_parser = self._get_qid_parser()

        self.max_per_qid = max_per_qid
        self.qid_occurrences = defaultdict(int)

    def __iter__(self):
        for file in sorted(list(self.damuel_path.iterdir())):
            if (
                file.is_file()
                and file.name.startswith("part")
                and (self.filename_is_ok is None or self.filename_is_ok(file.name))
            ):
                print(os.getpid(), file, flush=True)
                if file.name.endswith(".xz"):
                    with lzma.open(file, "rt") as f:
                        for mention in self._iterate_file(f):
                            yield mention
                else:
                    with file.open("r") as f:
                        for mention in self._iterate_file(f):
                            yield mention

    def _qid_to_int(self, qid):
        return int(qid[1:])

    def _qid_to_str(self, qid):
        return qid

    def _get_qid_parser(self):
        if self.treat_qids_as_ints:
            return self._qid_to_int
        else:
            return self._qid_to_str

    def _iterate_file(self, f):
        for line in f:  # iter(file_map.readline, b""):
            # damuel_entry = json.loads(line.decode("utf-8"))
            damuel_entry = json.loads(line)

            if self.only_wiki and "wiki" not in damuel_entry:
                continue

            # label = damuel_entry["label"]
            label = damuel_entry["wiki"]["title"]
            description = damuel_entry["wiki"]["text"]
            qid = self.qid_parser(damuel_entry["qid"])

            if self.qid_occurrences[qid] >= self.max_per_qid:
                continue
            self.qid_occurrences[qid] += 1

            class_enhanced_description = (
                f"{self.class_token} {label} {self.class_token} {description}"
            )

            description_tokens = self.tokenizer(
                class_enhanced_description,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=self.expected_size,
            )
            assert len(description_tokens["input_ids"]) <= self.expected_size

            description_pair = MentionQidPair(description_tokens, qid)

            yield description_pair
