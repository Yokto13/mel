from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np

from utils.loaders import load_mentions


class Mixer:
    """Gets directory with many tokens and qids files, and buffer size.
    It mixes the content of the files leaving the number of the same.

    It works by chunking all the files into groups <= buffer size, loading their content, shuffling it and writing it back to the same files.
    """

    def __init__(self, buffer_size: int = 10) -> None:
        self.buffer_size = buffer_size

    def mix(self, file_paths: list[Path], n_of_mixings: int) -> None:
        file_paths = deepcopy(file_paths)
        for _ in range(n_of_mixings):
            self._mix(file_paths)

    def _mix(self, file_paths: list[Path]) -> None:
        np.random.shuffle(file_paths)
        for chunk in self._chunk(file_paths, self.buffer_size):
            tokens, qids = self._load_tokens_and_qids(chunk)
            tokens, qids = self._shuffle(tokens, qids)
            self._save_tokens_and_qids(tokens, qids, chunk)

    def _shuffle(
        self, tokens: np.ndarray, qids: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        p = np.random.permutation(len(tokens))

        return tokens[p], qids[p]

    def _save_tokens_and_qids(
        self, tokens: np.ndarray, qids: np.ndarray, chunk: list[Path]
    ) -> None:
        token_chunk_size = len(tokens) // len(chunk)

        tokens_chunked = self._chunk(tokens, token_chunk_size)
        qids_chunked = self._chunk(qids, token_chunk_size)

        for tokens, qids, file_path in zip(tokens_chunked, qids_chunked, chunk):
            np.savez_compressed(file_path, tokens=tokens, qids=qids)

    def _load_tokens_and_qids(self, chunk: list[Path]) -> tuple[np.ndarray, np.ndarray]:
        all_tokens = []
        all_qids = []
        for file_path in chunk:
            tokens, qids = load_mentions(file_path)
            all_tokens.append(tokens)
            all_qids.append(qids)

        all_tokens = np.concatenate(all_tokens)
        all_qids = np.concatenate(all_qids)

        return all_tokens, all_qids

    def _chunk(self, data: list[Any], chunk_size: int) -> list[list[Any]]:
        return [data[i : i + chunk_size] for i in range(0, len(data), chunk_size)]
