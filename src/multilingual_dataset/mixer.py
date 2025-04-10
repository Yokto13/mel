import concurrent.futures
from collections.abc import Iterator
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
            if i == n_of_mixings - 1 and compress_output:
                self._mix(file_paths, compress_output=True)
            else:
                self._mix(file_paths, compress_output=False)

    def _mix(self, file_paths: list[Path], compress_output: bool) -> None:
        np.random.shuffle(file_paths)
        for chunk in tqdm(
            self._chunk(file_paths, self.buffer_size),
            desc="Mixing",
            total=len(file_paths) // self.buffer_size + 1,
        ):
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

        def save_file(tokens, qids, file_path):
            if compress:
                np.savez_compressed(file_path, tokens=tokens, qids=qids)
            else:
                np.savez(file_path, tokens=tokens, qids=qids)

        with concurrent.futures.ThreadPoolExecutor(max_workers=100) as executor:
            futures = [
                executor.submit(save_file, tokens, qids, file_path)
                for tokens, qids, file_path in zip(tokens_chunked, qids_chunked, chunk)
            ]
            concurrent.futures.wait(futures)

    def _get_tokens_qids_chunks(
        self, tokens: np.ndarray, qids: np.ndarray, chunk_count: int
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        token_chunk_size = len(tokens) // chunk_count
        return self._chunk(tokens, token_chunk_size), self._chunk(
            qids, token_chunk_size
        )

    def _load_tokens_and_qids(self, chunk: list[Path]) -> tuple[np.ndarray, np.ndarray]:
        def load_file(file_path):
            return load_mentions(file_path)

        with concurrent.futures.ThreadPoolExecutor(max_workers=100) as executor:
            results = list(executor.map(load_file, chunk))

        all_tokens, all_qids = zip(*results)

        all_tokens = np.concatenate(all_tokens)
        all_qids = np.concatenate(all_qids)

        return all_tokens, all_qids

    def _chunk(self, data: list[Any], chunk_size: int) -> Iterator[list[Any]]:
        for i in range(0, len(data), chunk_size):
            yield data[i : i + chunk_size]
