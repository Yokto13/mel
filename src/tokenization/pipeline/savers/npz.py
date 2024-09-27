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

        tokens_array = np.array(tokens_list)
        qids_array = np.array(qids_list)

        if self.compress:
            np.savez_compressed(self.filename, tokens=tokens_array, qids=qids_array)
        else:
            np.savez(self.filename, tokens=tokens_array, qids=qids_array)

        yield  # To comply with the generator interface
