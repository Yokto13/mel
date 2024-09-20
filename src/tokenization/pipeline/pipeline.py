import os
import json
import lzma
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from collections.abc import Generator, Callable
from tqdm import tqdm
from pathlib import Path

from data_processors.tokens.tokenizer_wrapper import TokenizerWrapper


def contains_wiki_key(entry: dict) -> bool:
    """Returns True if the entry contains the 'wiki' key."""
    return "wiki" in entry


class TokenizationStep(ABC):
    @abstractmethod
    def process(
        self, input_gen: Generator[str, None, None] = None
    ) -> Generator[str, None, None]:
        pass


class TokenizationPipeline:
    def __init__(self):
        self.steps: list[TokenizationStep] = []

    def add(self, step: TokenizationStep) -> None:
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


class LoaderStep(TokenizationStep, ABC):
    def __init__(self, path: str):
        self.path = path

    @abstractmethod
    def process(self) -> Generator[str, None, None]:
        pass


class DaMuELLoader(LoaderStep):
    def __init__(self, path: str):
        super().__init__(path)
        if not os.path.isdir(self.path):
            raise ValueError(f"Provided path {self.path} is not a directory")

    def process(self) -> Generator[str, None, None]:
        for filename in os.listdir(self.path):
            if filename.startswith("part-"):
                file_path = os.path.join(self.path, filename)
                with self._open_file(file_path) as file:
                    for line in file:
                        yield json.loads(line)

    def _open_file(self, file_path: str):
        if file_path.endswith(".xz"):
            return lzma.open(file_path, "rt")
        else:
            return open(file_path, "r")


class DaMuELFilter(TokenizationStep):
    def __init__(self, filter_func: Callable[[dict], bool]):
        self.filter_func = filter_func

    def process(
        self, input_gen: Generator[dict, None, None]
    ) -> Generator[dict, None, None]:
        for json_obj in input_gen:
            if self.filter_func(json_obj):
                yield json_obj


class DaMuELLinkProcessor(TokenizationStep):
    def process(
        self, input_gen: Generator[dict, None, None]
    ) -> Generator[str, None, None]:
        for json_obj in input_gen:
            # Implement DaMuEL link processing logic here
            yield f"DaMuEL links processed: {json_obj}"


class DaMuELDescriptionProcessor(TokenizationStep):
    def process(
        self, input_gen: Generator[dict, None, None]
    ) -> Generator[str, None, None]:
        for json_obj in input_gen:
            # Implement DaMuEL description processing logic here
            yield f"DaMuEL descriptions processed: {json_obj}"


class MewsliLoader(LoaderStep):
    def __init__(self, mewsli_tsv_path: str, use_context: bool = True):
        super().__init__(mewsli_tsv_path)
        self.use_context = use_context
        self.data_df = pd.read_csv(mewsli_tsv_path, sep="\t")

    def process(self) -> Generator[tuple, None, None]:
        for row in self.data_df.itertuples():
            qid = int(row.qid[1:])
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


class ContextTokenizer(TokenizationStep):
    def process(
        self, input_gen: Generator[str, None, None]
    ) -> Generator[str, None, None]:
        for text in input_gen:
            # Implement context tokenization logic here
            yield f"Context tokenized: {text}"


class MentionOnlyTokenizer(TokenizationStep):
    def __init__(self, tokenizer, expected_size):
        self.tokenizer_wrapper = TokenizerWrapper(tokenizer, expected_size)

    def process(
        self, input_gen: Generator[tuple, None, None]
    ) -> Generator[tuple, None, None]:
        for mention, qid in input_gen:
            tokens = self.tokenizer_wrapper.tokenize(mention)
            yield tokens, qid


class NPZSaver(TokenizationStep):
    def __init__(self, filename: str, compress: bool = False):
        self.filename = filename
        self.compress = compress

    def process(
        self, input_gen: Generator[tuple, None, None]
    ) -> Generator[None, None, None]:
        tokens_list = []
        qids_list = []
        for tokens, qids in tqdm(input_gen):
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
        compress: bool = False,
    ):
        super().__init__()
        self.add(MewsliLoader(mewsli_tsv_path, use_context=False))
        self.add(MentionOnlyTokenizer(tokenizer, expected_size))
        self.add(NPZSaver(output_filename, compress))
