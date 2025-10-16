from pathlib import Path

import fire
import numpy as np
from scipy.sparse import csr_matrix
from tqdm import tqdm

from utils.loaders import map_qids_to_token_matrix


def update_tokens_in_file(file_path: Path, qid_to_new_tokens: csr_matrix) -> None:
    """Updates tokens in a single .npz file and returns if it was modified."""
    with np.load(file_path) as data:
        tokens = data["description_tokens"]
        qids = data["qids"]
        save_data = dict(data)

    for i, qid in enumerate(qids):
        if qid_to_new_tokens[qid].nnz > 0:
            tokens[i] = qid_to_new_tokens[qid].toarray()[0]

    save_data["description_tokens"] = tokens
    np.savez(file_path, **save_data)


def process_directory(dataset_dir: Path, new_tokens_dir: Path) -> None:
    """Orchestrates the token update process for an entire directory."""
    qid_to_new_tokens = map_qids_to_token_matrix(new_tokens_dir, verbose=True)
    files_to_process = list(dataset_dir.glob("*.npz"))

    for fp in tqdm(files_to_process, desc="Updating files"):
        update_tokens_in_file(fp, qid_to_new_tokens)


def main(dataset_dir, new_tokens_dir):
    dataset_dir = Path(dataset_dir)
    new_tokens_dir = Path(new_tokens_dir)
    print("Starting token update process...")
    process_directory(dataset_dir, new_tokens_dir)
    print("Update process completed.")


if __name__ == "__main__":
    fire.Fire(main)
