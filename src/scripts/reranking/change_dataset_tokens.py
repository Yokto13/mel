from pathlib import Path

import fire
import numpy as np
from scipy.sparse import csr_matrix
from tqdm import tqdm

from utils.loaders import map_qids_to_token_matrix


def update_tokens_in_file(file_path: Path, qid_to_new_tokens: csr_matrix) -> None:
    """Updates tokens in a single .npz file in place.

    Replaces each entry in "description_tokens" with the corresponding row from
    qid_to_new_tokens indexed by the file's qids. Raises ValueError if a qid has
    no corresponding tokens in qid_to_new_tokens.
    """
    print(file_path)
    with np.load(file_path) as data:
        tokens = data["description_tokens"]
        qids = data["qids"]
        save_data = dict(data)

    # tokens = np.empty((tokens.shape[0], qid_to_new_tokens.shape[1]), dtype=tokens.dtype)

    tokens = qid_to_new_tokens[qids].toarray()

    # for i, qid in enumerate(qids):
    #     if qid_to_new_tokens[qid].nnz > 0:
    #         tokens[i] = qid_to_new_tokens[qid].toarray()[0]
    #     else:
    #         # We could also pad/truncate here if needed but this code should not really happen.
    #         raise ValueError(f"No new tokens found for qid {qid}")

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
