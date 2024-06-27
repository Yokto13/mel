import os
import multiprocessing

import lzma


def get_cpus_available():
    cpus = os.environ.get("SLURM_JOB_CPUS_PER_NODE")
    return int(cpus) if cpus is not None else 1


def decompress_pickled_file(file_path):
    with open(file_path, "rb") as f:
        with lzma.open(f) as xz_f:
            with open(file_path.replace(".xz", ""), "wb") as out_f:
                out_f.write(xz_f.read())
    os.remove(file_path)


def decompress_files_in_directory(directory_path):
    xz_files = [
        os.path.join(directory_path, f)
        for f in os.listdir(directory_path)
        if f.endswith(".xz")
    ]

    with multiprocessing.Pool(processes=get_cpus_available()) as pool:
        pool.map(decompress_pickled_file, xz_files)
