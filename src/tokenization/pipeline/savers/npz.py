from collections.abc import Generator

import numpy as np

from ..base import PipelineStep


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

        self._save_data(tokens_list, qids_list)
        yield  # To comply with the generator interface

    def _save_data(self, tokens_list: list, qids_list: list) -> None:
        filename = self.get_current_filename()
        tokens_array = np.array(tokens_list)
        qids_array = np.array(qids_list)
        if self.compress:
            np.savez_compressed(filename, tokens=tokens_array, qids=qids_array)
        else:
            np.savez(filename, tokens=tokens_array, qids=qids_array)

    def get_current_filename(self) -> str:
        return self.filename


class NPZSaverIncremental(NPZSaver):
    def __init__(self, filename: str, compress: bool = False, save_every: int = 10000):
        self.filename = filename
        self.compress = compress
        self.save_every = save_every
        self.counter = 0

    def process(
        self, input_gen: Generator[tuple, None, None]
    ) -> Generator[None, None, None]:
        tokens_list = []
        qids_list = []
        for tokens, qids in input_gen:
            tokens_list.append(tokens)
            qids_list.append(qids)

            if len(tokens_list) >= self.save_every:
                self._save_data(tokens_list, qids_list)
                tokens_list = []
                qids_list = []

        if len(tokens_list) > 0:
            self._save_data(tokens_list, qids_list)

        yield  # To comply with the generator interface

    def get_current_filename(self) -> str:
        if "npz" in self.filename:
            filename = self.filename[:-4]
        else:
            filename = self.filename
        self.counter += 1
        return f"{filename}_{self.counter}.npz"
