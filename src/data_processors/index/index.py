from collections import defaultdict
import lzma
from math import inf
from pathlib import Path
import pickle

import sys

sys.path.append("..")

import numpy as np
import torch
import scann


class Index:
    """Wrapper around scann for easier build"""

    def __init__(self, embs, qids, default_index_build=True):
        self.embs = embs

        self.embs = self.embs / np.linalg.norm(self.embs, axis=1, keepdims=True)

        self.qids = qids.astype(int)
        self.scann_index = None

        if default_index_build:
            self.build_index()

    def __contains__(self, qid):
        return qid in self._qids_set

    @classmethod
    def from_dir(cls, path, max_per_qid=inf):
        qid_counter = defaultdict(int)

        if isinstance(path, str):
            path = Path(path)
        all_embs, all_qids = [], []

        for fn_embs in sorted(path.iterdir()):
            if not cls.is_emb_file(fn_embs):
                continue
            print("Loading", fn_embs.name)
            hash_str = cls.extract_hash(fn_embs)
            embs_fn = path / f"embs_{hash_str}.npy"
            qids_fn = path / f"qids_{hash_str}.npy"

            embs = np.load(embs_fn)
            qids = np.load(qids_fn)

            embs, qids = cls.filter_based_on_max_per_qid(
                embs, qids, max_per_qid, qid_counter
            )

            all_embs.append(embs)
            all_qids.append(qids)

        all_embs = np.concatenate(all_embs)
        all_qids = np.concatenate(all_qids)

        return cls(all_embs, all_qids)

    @classmethod
    def from_iterable_and_model(cls, dataloader, model, max_per_qid=inf):
        all_embs, all_qids, all_return_embs = [], [], []
        qid_counter = defaultdict(int)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        for orig_embs, orig_qids in dataloader:
            embs, qids = cls.filter_based_on_max_per_qid(
                orig_embs.numpy(), orig_qids.numpy(), max_per_qid, qid_counter
            )

            if len(embs) == 0:
                continue

            all_return_embs.append(embs)

            embs = torch.tensor(embs, dtype=torch.float32).to(device)

            new_embs = model.forward_only_embeddings(embs).cpu().detach().numpy()

            new_embs = new_embs.astype(np.float16)
            all_embs.append(new_embs)
            all_qids.append(qids)

        all_embs = np.concatenate(all_embs)
        all_qids = np.concatenate(all_qids)
        return cls(all_embs, all_qids)

    @classmethod
    def is_emb_file(cls, fn):
        return fn.suffix == ".npy" and fn.name.startswith("embs")

    @classmethod
    def extract_hash(cls, fn):
        return fn.name.split("_")[-1].split(".")[0]

    @classmethod
    def filter_based_on_max_per_qid(
        cls, embs, qids, max_per_qid, ref_on_qid_counter, return_embs=None
    ):
        filtered_qids, filtered_embs, filtered_return_embs = [], [], []
        for i in range(len(qids)):
            if ref_on_qid_counter[qids[i]] < max_per_qid:
                ref_on_qid_counter[qids[i]] += 1
                filtered_qids.append(qids[i])
                filtered_embs.append(embs[i])
                if return_embs is not None:
                    filtered_return_embs.append(return_embs[i])
        if return_embs is not None:
            return (
                np.array(filtered_embs),
                np.array(filtered_qids),
                np.array(filtered_return_embs),
            )
        return np.array(filtered_embs), np.array(filtered_qids)

    def build_index(
        self,
        num_leaves=5000,
        num_leaves_to_search=100,
        training_sample_size=10**6,
        reordering_size=250,
        use_assymetric_hashing=True,
    ):
        training_sample_size = int(min(0.5 * len(self.embs), training_sample_size))
        num_leaves = min(num_leaves, training_sample_size)
        n_of_clusters = min(
            training_sample_size, 100
        )  # so we can test with tiny datasets
        builder = scann.scann_ops_pybind.builder(
            self.embs, n_of_clusters, "dot_product"
        ).tree(
            num_leaves=num_leaves,
            num_leaves_to_search=num_leaves_to_search,
            training_sample_size=training_sample_size,
        )
        if use_assymetric_hashing:
            builder = builder.score_ah(2, anisotropic_quantization_threshold=0.2)
        else:
            builder = builder.score_brute_force(quantize=True)
        self.scann_index = builder.reorder(reordering_size).build()

    @property
    def index(self):
        if self.scann_index is None:
            raise ValueError("Index not built")
        return self.scann_index

    def __len__(self):
        return len(self.embs)
