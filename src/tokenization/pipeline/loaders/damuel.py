import bz2
import lzma
import os
from collections.abc import Generator

import orjson
from tqdm.auto import tqdm

from ..base import Pipeline, PipelineStep
from ..filters import WikiKeyFilter
from .base import LoaderStep
from .qid_parsing import parse_qid


class DaMuELStartLoader(LoaderStep):
    def __init__(self, path: str, remainder: int = None, mod: int = None):
        super().__init__(path)
        if not os.path.isdir(self.path):
            raise ValueError(f"Provided path {self.path} is not a directory")
        self.remainder = remainder
        self.mod = mod

    def process(
        self, input_gen: Generator[str, None, None] = None
    ) -> Generator[str, None, None]:
        if input_gen is not None:
            raise ValueError("DaMuELStartLoader does not support input generator")
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
        elif file_path.endswith(".bz2"):
            return bz2.open(file_path, "rt")
        else:
            return open(file_path, "r")

    def _should_process_file(self, filename: str) -> bool:
        if self.remainder is None or self.mod is None:
            return True
        if filename.endswith(".xz"):
            filename = filename[:-3]
        elif filename.endswith(".bz2"):
            filename = filename[:-4]
        file_number = int(filename.split("-")[1])
        return file_number % self.mod == self.remainder


class DaMuELLinkProcessor(PipelineStep):
    def __init__(
        self,
        use_context: bool = False,
        require_wiki_origin: bool = True,
    ):
        super().__init__()
        self.use_context = use_context
        self.require_wiki_origin = require_wiki_origin

    def process(
        self, input_gen: Generator[dict, None, None]
    ) -> Generator[tuple, None, None]:
        for damuel_entry in input_gen:
            if "wiki" not in damuel_entry:
                continue
            wiki = damuel_entry["wiki"]
            links = (link for link in wiki["links"] if not self._should_skip_link(link))
            for link in links:
                yield from self._process_link(wiki, link)

    def _process_link(self, wiki: dict, link: dict) -> Generator[tuple, None, None]:
        qid = parse_qid(link["qid"])
        start = link["start"]
        end = link["end"] - 1
        try:
            mention_slice_chars = slice(
                wiki["tokens"][start]["start"], wiki["tokens"][end]["end"]
            )
        except IndexError:
            print("Index Error, skipping")
            return
        if self.use_context:
            yield mention_slice_chars, wiki["text"], qid
        else:
            yield wiki["text"][mention_slice_chars], qid

    def _should_skip_link(self, link: dict) -> bool:
        if "qid" not in link:
            return True
        if self.require_wiki_origin and link["origin"] != "wiki":
            return True
        return False


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
            original_title = title
            title_wrapped = self._wrap_title(title, self.label_token)

            description = self._extract_description(damuel_entry)
            if description is None:
                description = ""
            text = self.construct_text_from_title_and_description(
                title_wrapped, description, original_title
            )

            qid = parse_qid(damuel_entry["qid"])
            yield text, qid

    def _process_without_context(
        self, input_gen: Generator[dict, None, None]
    ) -> Generator[tuple, None, None]:
        for damuel_entry in input_gen:
            title = self._extract_title(damuel_entry)
            if title is None:
                continue

            qid = parse_qid(damuel_entry["qid"])
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

    @classmethod
    def construct_text_from_title_and_description(
        cls, title: str, description: str, original_title: str | None = None
    ) -> str:
        if original_title is not None and description.strip().startswith(
            original_title.strip()
        ):
            return f"{title}\n{description[len(original_title):]}"
        return f"{title}\n{description}"


class DaMuELDescriptionLoader(Pipeline):
    def __init__(
        self,
        path: str,
        require_wiki_page: bool,
        remainder: int = None,
        mod: int = None,
        use_context: bool = False,
        label_token: str = None,
    ):
        super().__init__()
        self.add(DaMuELStartLoader(path, remainder, mod))
        if require_wiki_page:
            self.add(WikiKeyFilter())
        self.add(DaMuELDescriptionProcessor(use_context, label_token))


class DaMuELLinkLoader(Pipeline):
    def __init__(
        self,
        path: str,
        remainder: int = None,
        mod: int = None,
        use_context: bool = False,
        require_link_wiki_origin: bool = True,
    ):
        super().__init__()
        self.add(DaMuELStartLoader(path, remainder, mod))
        # Here WikiKeyFilter is required because links are only in Wikipages
        self.add(WikiKeyFilter())
        self.add(DaMuELLinkProcessor(use_context, require_link_wiki_origin))
