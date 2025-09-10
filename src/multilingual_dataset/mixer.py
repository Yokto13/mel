import concurrent.futures
import multiprocessing as mp
from collections.abc import Iterator
from copy import deepcopy
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from utils.loaders import load_mentions


def _shuffle(tokens: np.ndarray, qids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Shuffle tokens and qids arrays using the same permutation."""
    p = np.random.permutation(len(tokens))
    return tokens[p], qids[p]


def _load_tokens_and_qids(chunk: list[Path]) -> tuple[np.ndarray, np.ndarray]:
    """Load tokens and qids from a chunk of files."""

    def load_file(file_path):
        return load_mentions(file_path)

    with concurrent.futures.ThreadPoolExecutor(max_workers=100) as executor:
        results = list(executor.map(load_file, chunk))

    all_tokens, all_qids = zip(*results)
    all_tokens = np.concatenate(all_tokens)
    all_qids = np.concatenate(all_qids)

    return all_tokens, all_qids


def _save_tokens_and_qids(
    tokens: np.ndarray,
    qids: np.ndarray,
    chunk: list[Path],
    compress: bool = True,
) -> None:
    """Save tokens and qids back to their respective files."""

    def _chunk_arrays(data: np.ndarray, chunk_size: int) -> list[np.ndarray]:
        """Chunk a numpy array into smaller arrays."""
        return [data[i : i + chunk_size] for i in range(0, len(data), chunk_size)]

    token_chunk_size = len(tokens) // len(chunk)
    tokens_chunked = _chunk_arrays(tokens, token_chunk_size)
    qids_chunked = _chunk_arrays(qids, token_chunk_size)

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


def mix_chunk(chunk: list[Path], compress_output: bool) -> None:
    """Mix the tokens and qids for a single chunk of files."""
    tokens, qids = _load_tokens_and_qids(chunk)
    tokens, qids = _shuffle(tokens, qids)
    _save_tokens_and_qids(tokens, qids, chunk, compress_output)


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
            mix_chunk(chunk, compress_output)

    def _chunk(self, data: list[Any], chunk_size: int) -> Iterator[list[Any]]:
        for i in range(0, len(data), chunk_size):
            yield data[i : i + chunk_size]


class ParallelMixer(Mixer):
    """Parallel version of the Mixer class that uses multiprocessing for mixing files.

    This class should be faster than the original Mixer class but requires more memory.
    """

    def __init__(self, buffer_size: int = 10, n_workers: int = 4) -> None:
        super().__init__(buffer_size)
        self.n_workers = n_workers

    def _mix(self, file_paths: list[Path], compress_output: bool) -> None:
        np.random.shuffle(file_paths)
        chunks = list(self._chunk(file_paths, self.buffer_size))

        process_chunk_with_compression = partial(
            mix_chunk, compress_output=compress_output
        )

        with mp.Pool(processes=self.n_workers) as pool:
            list(
                tqdm(
                    pool.imap(process_chunk_with_compression, chunks),
                    total=len(chunks),
                    desc="Parallel Mixing",
                )
            )
