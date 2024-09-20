import os
import orjson
import lzma
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from collections.abc import Generator, Callable
from typing import Any
from tqdm import tqdm
from pathlib import Path

from data_processors.tokens.tokenizer_wrapper import TokenizerWrapper
from data_processors.tokens.tokens_cutter import TokensCutter, TokensCutterV2


def contains_wiki_key(entry: dict) -> bool:
    """Returns True if the entry contains the 'wiki' key."""
    return "wiki" in entry


class PipelineStep(ABC):
    @abstractmethod
    def process(
        self, input_gen: Generator[str, None, None] = None
    ) -> Generator[str, None, None]:
        pass


class TokenizationPipeline:
    def __init__(self):
        self.steps: list[PipelineStep] = []

    def add(self, step: PipelineStep) -> None:
        self.steps.append(step)

    def run(self) -> None:
        def generator_chain() -> Generator[str, None, None]:
            gen = self.steps[0].process()  # Initial generator from the first step
            for step in self.steps[1:]:
                gen = step.process(gen)
            yield from gen

        # Consume the final generator to ensure all steps are executed
        # The generator_chain function creates a chain of generators, where each step's generator
        # processes the output of the previous step's generator. The final generator is consumed
        # by iterating over it with a for loop.
        for _ in generator_chain():
            pass

    def __str__(self) -> str:
        return "\n".join(
            [
                "Tokenization Pipeline Steps:",
                *[
                    f"{i}. {step.__class__.__name__}"
                    for i, step in enumerate(self.steps, 1)
                ],
            ]
        )


class LoaderStep(PipelineStep, ABC):
    def __init__(self, path: str):
        self.path = path

    @abstractmethod
    def process(self) -> Generator[str, None, None]:
        pass


class DaMuELLoader(LoaderStep):
    def __init__(self, path: str, remainder: int = None, mod: int = None):
        super().__init__(path)
        if not os.path.isdir(self.path):
            raise ValueError(f"Provided path {self.path} is not a directory")
        self.remainder = remainder
        self.mod = mod

    def process(self) -> Generator[str, None, None]:
        file_list = [
            filename
            for filename in os.listdir(self.path)
            if filename.startswith("part-")
        ]

        if self.mod is not None:
            file_list = [
                filename
                for filename in file_list
                if self._should_process_file(filename)
            ]

        tqdm_position = self.remainder if self.remainder is not None else 0
        tqdm_desc = f"Processing DaMuEL files {self.path[-6:]}"
        if self.mod is not None:
            tqdm_desc += f" Remainder: {self.remainder}, Mod: {self.mod}"
        for filename in tqdm(
            file_list,
            desc=tqdm_desc,
            position=tqdm_position,
        ):
            file_path = os.path.join(self.path, filename)
            with self._open_file(file_path) as file:
                for line in file:
                    yield orjson.loads(line)

    def _open_file(self, file_path: str):
        if file_path.endswith(".xz"):
            return lzma.open(file_path, "rt")
        else:
            return open(file_path, "r")

    def _should_process_file(self, filename: str) -> bool:
        if self.remainder is None or self.mod is None:
            return True
        if filename.endswith(".xz"):
            filename = filename[:-3]
        file_number = int(filename.split("-")[1])
        return file_number % self.mod == self.remainder


class Filter(PipelineStep):
    def __init__(self, filter_func: Callable[[dict], bool]):
        self.filter_func = filter_func

    def process(
        self, input_gen: Generator[Any, None, None]
    ) -> Generator[dict, None, None]:
        for obj in input_gen:
            if self.filter_func(obj):
                yield obj


class DaMuELLinkProcessor(PipelineStep):
    def __init__(
        self,
        tokenizer,
        expected_size,
        require_wiki_origin,
        use_context: bool = False,
        label_token: str = None,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.expected_size = expected_size
        self.require_wiki_origin = require_wiki_origin
        self.use_context = use_context
        self.label_token = label_token
        if use_context and label_token is None:
            raise ValueError("Label token must be provided for context mode")

    def process(
        self, input_gen: Generator[dict, None, None]
    ) -> Generator[str, None, None]:
        if self.use_context:
            yield from self._process_with_context(input_gen)
        else:
            yield from self._process_without_context(input_gen)

    def _process_with_context(
        self, input_gen: Generator[dict, None, None]
    ) -> Generator[str, None, None]:
        for damuel_entry in input_gen:
            wiki = damuel_entry["wiki"]  # requires filter prior to this
            damuel_tokens = wiki["tokens"]
            links = [link for link in wiki["links"] if not self._should_skip_link(link)]
            mention_slices = []
            choosen_links = []
            for link_idx, link in enumerate(links):
                start = link["start"]
                end = link["end"] - 1
                try:
                    mention_slice_chars = slice(
                        damuel_tokens[start]["start"], damuel_tokens[end]["end"]
                    )
                except IndexError:
                    print("Index Error, skipping")
                    continue
                mention_slices.append(mention_slice_chars)
                choosen_links.append(link)

            text_with_tokens_around_mentions = self._add_tokens_around_mentions(
                wiki["text"], mention_slices, self.label_token
            )

            tokens_cutter = TokensCutterV2(
                text_with_tokens_around_mentions,
                self.tokenizer,
                self.expected_size,
                self.label_token,
            )
            for link_idx, link in enumerate(choosen_links):
                qid = self._parse_qid(link["qid"])
                cutted_tokens = tokens_cutter.cut(link_idx)
                yield cutted_tokens, qid

    def _add_tokens_around_mentions(
        self, text: str, mention_slices: list[slice], label_token: str
    ) -> str:
        text_chunks = []
        prev_mention_end = 0
        for mention_slice in mention_slices:
            text_chunks.append(text[prev_mention_end : mention_slice.start])
            text_chunks.append(self.label_token)
            text_chunks.append(text[mention_slice])
            text_chunks.append(self.label_token)
            prev_mention_end = mention_slice.stop
        text_chunks.append(text[prev_mention_end:])
        return "".join(text_chunks)

    def _parse_qid(self, qid: str) -> int:
        return int(qid[1:])

    def _should_skip_link(self, link: dict) -> bool:
        if "qid" not in link:
            return True
        if self.require_wiki_origin and link["origin"] != "wiki":
            return True
        return False

    def _process_without_context(
        self, input_gen: Generator[dict, None, None]
    ) -> Generator[str, None, None]:
        raise NotImplementedError("Not implemented yet")


class DaMuELDescriptionProcessor(PipelineStep):
    def __init__(self, use_context: bool = False, label_token: str = None):
        super().__init__()
        self.use_context = use_context
        self.label_token = label_token
        if use_context and label_token is None:
            raise ValueError("Label token must be provided for context mode")

    def process(
        self, input_gen: Generator[dict, None, None]
    ) -> Generator[tuple, None, None]:
        if self.use_context:
            yield from self._process_with_context(input_gen)
        else:
            yield from self._process_without_context(input_gen)

    def _process_with_context(
        self, input_gen: Generator[dict, None, None]
    ) -> Generator[tuple, None, None]:
        for damuel_entry in input_gen:
            title = self._extract_title(damuel_entry)
            if title is None:
                continue
            title = self._wrap_title(title, self.label_token)

            description = self._extract_description(damuel_entry)
            if description is None:
                description = ""
            text = self._construct_text_from_title_and_description(title, description)

            qid = self._parse_qid(damuel_entry["qid"])
            yield text, qid

    def _process_without_context(
        self, input_gen: Generator[dict, None, None]
    ) -> Generator[tuple, None, None]:
        for damuel_entry in input_gen:
            title = self._extract_title(damuel_entry)
            if title is None:
                continue

            qid = self._parse_qid(damuel_entry["qid"])
            yield title, qid

    def _extract_title(self, damuel_entry: dict) -> str:
        if "wiki" in damuel_entry:
            return damuel_entry["wiki"]["title"]
        elif "label" in damuel_entry:
            return damuel_entry["label"]
        return None

    def _extract_description(self, damuel_entry: dict) -> str:
        if "wiki" in damuel_entry:
            return damuel_entry["wiki"]["text"]
        elif "description" in damuel_entry:
            return damuel_entry["description"]
        return None

    def _wrap_title(self, title: str, label_token: str) -> str:
        return f"{label_token}{title}{label_token}"

    def _construct_text_from_title_and_description(
        self, title: str, description: str
    ) -> str:
        return f"{title}\n{description}"

    def _parse_qid(self, qid: str) -> int:
        return int(qid[1:])


class MewsliLoader(LoaderStep):
    def __init__(self, mewsli_tsv_path: str, use_context: bool = True):
        super().__init__(mewsli_tsv_path)
        self.use_context = use_context
        self.data_df = pd.read_csv(mewsli_tsv_path, sep="\t")

    def process(self) -> Generator[tuple, None, None]:
        for row in tqdm(
            self.data_df.itertuples(),
            total=len(self.data_df),
            desc=f"Processing {self.path}",
        ):
            qid = self._parse_qid(row.qid)
            if self.use_context:
                with open(Path(self.path).parent / "text" / row.docid, "r") as f:
                    text = f.read()
                    mention_slice = self.get_mention_slice_from_row(row)
                    yield mention_slice, text, qid
            else:
                yield row.mention, qid

    def get_mention_slice_from_row(self, row):
        mention_start = row.position
        mention_end = row.position + row.length
        return slice(mention_start, mention_end)

    def _parse_qid(self, qid: str) -> int:
        return int(qid[1:])


class ContextTokenizer(PipelineStep):
    def __init__(self, tokenizer, expected_size):
        self.tokenizer_wrapper = TokenizerWrapper(tokenizer, expected_size)
        self.expected_size = expected_size

    def process(
        self, input_gen: Generator[tuple, None, None]
    ) -> Generator[tuple, None, None]:
        for mention_slice, text, qid in input_gen:
            tokens_cutter = TokensCutter(
                text, self.tokenizer_wrapper, self.expected_size
            )
            tokens = tokens_cutter.cut_mention_with_context(mention_slice)
            yield tokens, qid


class SimpleTokenizer(PipelineStep):
    def __init__(self, tokenizer, expected_size):
        self.tokenizer_wrapper = TokenizerWrapper(tokenizer, expected_size)

    def process(
        self, input_gen: Generator[tuple, None, None]
    ) -> Generator[tuple, None, None]:
        for mention, qid in input_gen:
            tokens = self.tokenizer_wrapper.tokenize(mention)
            yield tokens, qid


class NPZSaver(PipelineStep):
    def __init__(self, filename: str, compress: bool = False):
        self.filename = filename
        self.compress = compress

    def process(
        self, input_gen: Generator[tuple, None, None]
    ) -> Generator[None, None, None]:
        tokens_list = []
        qids_list = []
        for tokens, qids in input_gen:
            tokens_list.append(tokens)
            qids_list.append(qids)

        tokens_array = np.array(tokens_list)
        qids_array = np.array(qids_list)

        if self.compress:
            np.savez_compressed(self.filename, tokens=tokens_array, qids=qids_array)
        else:
            np.savez(self.filename, tokens=tokens_array, qids=qids_array)

        yield  # To comply with the generator interface


class MewsliMentionPipeline(TokenizationPipeline):
    def __init__(
        self,
        mewsli_tsv_path: str,
        tokenizer,
        expected_size: int,
        output_filename: str,
        compress: bool = True,
    ):
        super().__init__()
        self.add(MewsliLoader(mewsli_tsv_path, use_context=False))
        self.add(SimpleTokenizer(tokenizer, expected_size))
        self.add(NPZSaver(output_filename, compress))


class MewsliMentionContextPipeline(TokenizationPipeline):
    def __init__(
        self,
        mewsli_tsv_path: str,
        tokenizer,
        expected_size: int,
        output_filename: str,
        compress: bool = True,
    ):
        super().__init__()
        self.add(MewsliLoader(mewsli_tsv_path, use_context=True))
        self.add(ContextTokenizer(tokenizer, expected_size))
        self.add(NPZSaver(output_filename, compress))


class DamuelDescriptionMentionPipeline(TokenizationPipeline):
    def __init__(
        self,
        damuel_path: str,
        tokenizer,
        expected_size: int,
        output_filename: str,
        compress: bool = True,
        remainder: int = None,
        mod: int = None,
    ):
        super().__init__()
        self.add(DaMuELLoader(damuel_path, remainder, mod))
        self.add(Filter(contains_wiki_key))
        self.add(DaMuELDescriptionProcessor(use_context=False))
        self.add(SimpleTokenizer(tokenizer, expected_size))
        self.add(NPZSaver(output_filename, compress))


class DamuelDescriptionContextPipeline(TokenizationPipeline):
    def __init__(
        self,
        damuel_path: str,
        tokenizer,
        expected_size: int,
        output_filename: str,
        label_token: str,
        compress: bool = True,
        remainder: int = None,
        mod: int = None,
    ):
        super().__init__()
        self.add(DaMuELLoader(damuel_path, remainder, mod))
        self.add(Filter(contains_wiki_key))
        self.add(DaMuELDescriptionProcessor(use_context=True, label_token=label_token))
        self.add(SimpleTokenizer(tokenizer, expected_size))
        self.add(NPZSaver(output_filename, compress))


class DamuelLinkContextPipeline(TokenizationPipeline):
    def __init__(
        self,
        damuel_path: str,
        tokenizer,
        expected_size: int,
        output_filename: str,
        label_token: str,
        compress: bool = True,
        remainder: int = None,
        mod: int = None,
        require_wiki_origin: bool = True,
    ):
        super().__init__()
        self.add(DaMuELLoader(damuel_path, remainder, mod))
        self.add(Filter(contains_wiki_key))
        self.add(
            DaMuELLinkProcessor(
                tokenizer,
                expected_size,
                require_wiki_origin,
                use_context=True,
                label_token=label_token,
            )
        )
        self.add(NPZSaver(output_filename, compress))
