import os
import numpy as np
import concurrent.futures
import timeit


def read_qids_sequential(directory):
    qids = []
    for filename in os.listdir(directory):
        if filename.endswith(".npz"):
            file_path = os.path.join(directory, filename)
            data = np.load(file_path)
            qids.extend(data["qids"])
    return qids


def read_qids_parallel(directory):
    def read_file(filename):
        file_path = os.path.join(directory, filename)
        data = np.load(file_path)
        return data["qids"]

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for filename in os.listdir(directory):
            if filename.endswith(".npz"):
                futures.append(executor.submit(read_file, filename))

        qids = []
        for future in concurrent.futures.as_completed(futures):
            qids.extend(future.result())

    return qids


def read_file(filename):
    file_path = os.path.join(directory, filename)
    with np.load(file_path) as data:
        return data["qids"]


def read_qids_parallel_process(directory):
    with concurrent.futures.ProcessPoolExecutor() as executor:
        files = [f for f in os.listdir(directory) if f.endswith(".npz")]
        qids = list(executor.map(read_file, files))

    return [item for sublist in qids for item in sublist]


def benchmark_reading_methods(directory):
    sequential_time = timeit.timeit(lambda: read_qids_sequential(directory), number=1)
    parallel_time = timeit.timeit(lambda: read_qids_parallel(directory), number=1)
    parallel_process_time = timeit.timeit(
        lambda: read_qids_parallel_process(directory), number=1
    )
    print(f"Sequential reading time: {sequential_time:.2f} seconds")
    print(f"Parallel reading time: {parallel_time:.2f} seconds")
    print(f"Parallel process reading time: {parallel_process_time:.2f} seconds")
    speedup = sequential_time / parallel_time
    print(f"Parallel reading is {speedup:.2f} times faster than sequential reading")
    speedup = sequential_time / parallel_process_time
    print(
        f"Parallel process reading is {speedup:.2f} times faster than sequential reading"
    )


# Usage example
directory = "/lnet/work/home-students-external/farhan/troja/outputs/tokens_damuel_finetuning/en/links"
benchmark_reading_methods(directory)
