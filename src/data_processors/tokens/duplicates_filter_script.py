""" Map reduce like approach to remove duplicated QIDs, no longer used. """

from concurrent.futures import ThreadPoolExecutor
import lzma
import hashlib
import os
from pathlib import Path
import pickle
import subprocess
from subprocess import PIPE
import tempfile
from time import sleep

import numpy as np
from tqdm import tqdm
import wandb

from utils.argument_wrappers import ensure_datatypes


def can_start_process(running_processes, max_processes, paths_for_processing):
    return len(running_processes) < max_processes and len(paths_for_processing) > 0


def purge_processes(running_processes):
    for process in running_processes:
        if process.poll() is not None:
            print("Process finished")
            print(process.args)
            output, error = process.communicate()
            print("Output:", output)
            print("Error:", error)
            assert process.returncode == 0
            running_processes.remove(process)


def hash_str(s):
    return int(hashlib.sha1(s.encode()).hexdigest(), 16) % 10**8


def reduce_entries(dir):
    best_ils = {}
    for fn in os.listdir(dir):
        fp = os.path.join(dir, fn)
        print("Loading", fp)
        data = np.load(fp)
        for row in data:
            qid = row[0]
            weight = row[1]
            repre = (weight, row[2], row[3])
            if qid in best_ils:
                if weight > best_ils[qid][0]:
                    best_ils[qid] = repre
            else:
                best_ils[qid] = repre
    return best_ils


def filter_data(data, best_ils, file_path: Path):
    file_path_hash = hash_str(str(file_path))
    return [
        x
        for i, x in enumerate(data)
        if best_ils[x.qid][1] == i and best_ils[x.qid][2] == file_path_hash
    ]


def process_file(fp_and_best_ils):
    fp, best_ils = fp_and_best_ils
    print(fp)
    if not fp.is_file() or "xz" not in fp.suffix:
        return
    with lzma.open(fp, "rb") as f:
        data = pickle.load(f)
    data = filter_data(data, best_ils, fp)
    with lzma.open(fp, "wb") as f:
        pickle.dump(data, f)


@ensure_datatypes([Path, int], {})
def run_duplicates_filter_script(tokens_dir: Path, max_processes: int):
    wandb.init(
        project="filter-duplicates",
        config={
            "tokens_dir": tokens_dir,
            "max_processes": max_processes,
        },
    )
    temp_dir = tempfile.TemporaryDirectory()
    running_processes = []
    paths = [fp for fp in tokens_dir.iterdir() if fp.is_file() and "xz" in fp.suffix]
    # multiprocess map
    # for some reason torch breaks when used inside of multiprocess so I do it manually
    while len(paths) > 0 or len(running_processes) > 0:
        purge_processes(running_processes)
        if can_start_process(running_processes, max_processes, paths):
            process = subprocess.Popen(
                [
                    "bash",
                    "data_processors/tokens/duplicates_filter_process_file.sh",
                    str(paths.pop(-1)),
                    temp_dir.name,
                ],
                stderr=PIPE,
                stdout=PIPE,
            )
            running_processes.append(process)
        sleep(1)

    best_ils = reduce_entries(temp_dir.name)
    paths = [fp for fp in tokens_dir.iterdir() if fp.is_file() and "xz" in fp.suffix]
    temp_dir.cleanup()

    # We must be careful with the number of threads we use otherwise we OOM
    with ThreadPoolExecutor(max_workers=int(3 / 2 * max_processes)) as executor:
        for _ in tqdm(
            executor.map(process_file, ((fp, best_ils) for fp in paths)),
            total=len(paths),
        ):
            pass
