from abc import ABC, abstractmethod
import os
from pathlib import Path
from typing import Callable
import lzma


class DamuelIterator(ABC):
    def __init__(
        self,
        damuel_path,
        tokenizer,
        expected_size,
        filenumber_is_ok: Callable[[str], bool]
    ):
        self.damuel_path = damuel_path
        if not isinstance(damuel_path, Path):
            self.damuel_path = Path(damuel_path)
        self.expected_size = expected_size
        self.tokenizer = tokenizer
        self.filenumber_is_ok = filenumber_is_ok

    def __iter__(self):
        for file in sorted(list(self.damuel_path.iterdir())):
            if (
                file.is_file()
                and file.name.startswith("part")
                and self.filenumber_is_ok(int(file.name.split('-')[-1]))
            ):
                print(os.getpid(), file, flush=True)
                with self._get_file_obj(file) as f:
                    for dato in self._iterate_file(f):
                        yield dato

    def _get_file_obj(self, file):
        if file.name.endswith(".xz"):
            return lzma.open(file, "rt")
        else:
            return file.open("r")

    @abstractmethod
    def _iterate_file(self, f):
        pass


# proc je to tady ??????
class DamuelLinksIterator(DamuelIterator):
    def __init__(
        self,
        damuel_path,
        tokenizer,
        only_wiki=True,
        expected_size=64,
        filename_is_ok: Callable[[str], bool] = None,
        treat_qids_as_ints=True,
    ):
        super().__init__(
            damuel_path, tokenizer, expected_size, filename_is_ok, treat_qids_as_ints
        )
        self.only_wiki = only_wiki
