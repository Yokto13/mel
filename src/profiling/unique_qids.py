from timeit import timeit

import numba as nb
import numpy as np


@nb.njit
def create_unique_qid_index_numba(
    base_index: np.ndarray, qids: np.ndarray, batch_size: int
) -> np.ndarray:
    data_idx = np.empty(len(base_index), dtype=np.int64)
    qids_in_batch = np.empty(batch_size, dtype=qids.dtype)
    idx_counter = 0
    qids_counter = 0
    for i in range(len(base_index)):
        idx = base_index[i]
        qid = qids[idx]
        if qid not in qids_in_batch[:qids_counter]:
            data_idx[idx_counter] = idx
            idx_counter += 1
            qids_in_batch[qids_counter] = qid
            qids_counter += 1
        if qids_counter == batch_size:
            qids_counter = 0
    return data_idx[:idx_counter]


@nb.njit
def create_unique_qid_index_set(
    base_index: np.ndarray, qids: np.ndarray, batch_size: int
) -> np.ndarray:
    data_idx = np.empty(len(base_index), dtype=np.int64)
    qids_in_batch = set()
    idx_counter = 0
    for idx in base_index:
        qid = qids[idx]
        if qid not in qids_in_batch:
            data_idx[idx_counter] = idx
            idx_counter += 1
            qids_in_batch.add(qid)
            if len(qids_in_batch) == batch_size:
                qids_in_batch.clear()
    return data_idx[:idx_counter]


def generate_test_data(size: int, batch_size: int):
    base_index = np.arange(size)
    np.random.shuffle(base_index)
    qids = np.random.randint(0, size // 10, size=size)
    return base_index, qids, batch_size


def run_benchmark(func, base_index, qids, batch_size, number=50):
    return timeit(lambda: func(base_index, qids, batch_size), number=number)


if __name__ == "__main__":
    batch_sizes = [256, 800]
    sizes = [int(10**7)]

    for size in sizes:
        for batch_size in batch_sizes:
            base_index, qids, _ = generate_test_data(size, batch_size)

            # Warm-up for Numba
            create_unique_qid_index_numba(base_index, qids, batch_size)

            array_time = run_benchmark(
                create_unique_qid_index_numba, base_index, qids, batch_size
            )
            set_time = run_benchmark(
                create_unique_qid_index_set, base_index, qids, batch_size
            )

            print(f"Size: {size}, Batch Size: {batch_size}")
            print(f"Array-based time: {array_time:.6f} seconds")
            print(f"Set-based time: {set_time:.6f} seconds")
            print(f"Speedup: {set_time / array_time:.2f}x")
            print()
