from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from utils.loaders import load_mentions


class Mixer:
    """Gets directory with many tokens and qids files, and buffer size.
    It mixes the content of the files leaving the number of the same.

    It works by chunking all the files into groups <= buffer size, loading their content, shuffling it and writing it back to the same files.
    """

    def __init__(self, buffer_size: int = 10) -> None:
        self.buffer_size = buffer_size

    def mix(
        self, file_paths: list[Path], n_of_mixings: int = 10, compress_output=True
    ) -> None:
        print("mixxing")
        file_paths = deepcopy(file_paths)
        for i in tqdm(range(n_of_mixings)):
            if i == n_of_mixings - 1:
                self._mix(file_paths, compress_output)
            else:
                self._mix(file_paths, compress_output=False)

    def _mix(self, file_paths: list[Path], compress_output: bool) -> None:
        np.random.shuffle(file_paths)
        for chunk in self._chunk(file_paths, self.buffer_size):
            tokens, qids = self._load_tokens_and_qids(chunk)
            tokens, qids = self._shuffle(tokens, qids)
            self._save_tokens_and_qids(tokens, qids, chunk, compress_output)

    def _shuffle(
        self, tokens: np.ndarray, qids: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        p = np.random.permutation(len(tokens))

        return tokens[p], qids[p]

    def _save_tokens_and_qids(
        self,
        tokens: np.ndarray,
        qids: np.ndarray,
        chunk: list[Path],
        compress: bool = True,
    ) -> None:
        tokens_chunked, qids_chunked = self._get_tokens_qids_chunks(
            tokens, qids, len(chunk)
        )

        for tokens, qids, file_path in zip(tokens_chunked, qids_chunked, chunk):
            if compress:
                self._save_compressed(file_path, tokens, qids)
            else:
                self._save(file_path, tokens, qids)

    def _get_tokens_qids_chunks(
        self, tokens: np.ndarray, qids: np.ndarray, chunk_count: int
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        token_chunk_size = len(tokens) // chunk_count
        return self._chunk(tokens, token_chunk_size), self._chunk(
            qids, token_chunk_size
        )

    def _save_compressed(
        self, file_path: Path, tokens: np.ndarray, qids: np.ndarray
    ) -> None:
        np.savez_compressed(file_path, tokens=tokens, qids=qids)

    def _save(self, file_path: Path, tokens: np.ndarray, qids: np.ndarray) -> None:
        np.savez(file_path, tokens=tokens, qids=qids)

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
