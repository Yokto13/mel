from collections import defaultdict, Counter
from hashlib import sha1
from pathlib import Path

import numpy as np
from transformers import BertModel

from data_processors.index.index import Index


# TODO: Read unique_n and RecallCalculator and add tests.
def get_unique_n(iterable, n):
    seen = set()
    for i in iterable:
        if i not in seen:
            seen.add(i)
            yield i
        if len(seen) == n:
            break


class RecallCalculator:
    def __init__(self, scann_index, qids_in_index) -> None:
        self.scann_index = scann_index
        self.qids_in_index = qids_in_index

    def recall(self, mewsli_embs, mewsli_qids, k: int):
        qid_was_present = self._process_for_recall(mewsli_embs, mewsli_qids, k)
        return self._calculate_recall(qid_was_present)

    def _calculate_recall(self, qid_was_present):
        return sum(qid_was_present) / len(qid_was_present)

    def _get_neighboring_qids(self, queries_embs, k):
        qids_per_query = []
        neighbors, dists = self.scann_index.search_batched(
            queries_embs, final_num_neighbors=max(100000, k)
        )
        for ns in neighbors:
            ns_qids = self.qids_in_index[ns]
            unique_ns_qids = list(get_unique_n(ns_qids, k))
            qids_per_query.append(unique_ns_qids)
        return qids_per_query

    def _process_for_recall(self, mewsli_embs, mewsli_qids, k):
        qid_was_present = []
        for emb, qid in zip(mewsli_embs, mewsli_qids):
            negihboring_qids = self._get_neighboring_qids([emb], k)
            qid_was_present.append(qid in negihboring_qids[0])
        return qid_was_present


class OLPEAT:
    def __init__(self, train_dir, model_path) -> None:
        self.train_dir = train_dir
        if isinstance(self.train_dir, str):
            self.train_dir = Path(self.train_dir)
        self.model = BertModel.from_pretrained(model_path)
        self.max_R = 10
        self.recall_calculator = None
        self.embs_loader = Cacher()

    def train(self, gpus_available=None, batch_size=None):
        """
        Trains OLPEAT.

        Includes:
            - loads train tokens
            - embeds
            - builds searcher

        Note:
            At least one of gpus_available or batch_size should be None.
            If gpus_available is none None batch_size is calculated based on their number.
            If both are none, default batch size of 2**14 is used.
        """
        if batch_size is None and gpus_available is not None:
            batch_size = self._bs_simple_heuristic(gpus_available)
        if batch_size is None:
            embs, qids = self.embs_loader.get_embs_and_qids(self.train_dir, self.model)
        else:
            embs, qids = self.embs_loader.get_embs_and_qids(
                self.train_dir, self.model, batch_size
            )
        embs = self._normalize(embs)
        embs10, qids10 = self._filter_duplicates(embs, qids, 10)
        embs1, qids1 = self._filter_duplicates(embs, qids, 1)
        self.recall_calculator_at10 = RecallCalculator(
            Index(embs10, qids10).scann_index, qids10
        )
        self.recall_calculator_at1 = RecallCalculator(
            Index(embs1, qids1).scann_index, qids1
        )

    def recall_at(self, R, test_dir):
        assert R <= self.max_R
        embs, qids = self.embs_loader.get_embs_and_qids(test_dir, self.model)
        if R == 1:
            return self.recall_calculator_at1.recall(embs, qids, 1)
        if R == 10:
            return self.recall_calculator_at10.recall(embs, qids, 10)

    def _bs_simple_heuristic(self, gpus_available):
        return int(2**13 * gpus_available * 1.1)

    def _normalize(self, vecs):
        return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)

    def _filter_duplicates(self, embs, qids, R):
        """Hashes underlying embs bytes to find the same embs and filters the repeated once based on self.max_R."""
        if not embs.flags["C_CONTIGUOUS"]:
            embs = np.ascontiguousarray(embs)

        emb_qid_d = defaultdict(Counter)
        for emb, qid in zip(embs, qids):
            emb_qid_d[sha1(emb.tobytes()).hexdigest()][qid] += 1

        # keep only top R qids per emb
        for emb_hash, qid_counter in emb_qid_d.items():
            emb_qid_d[emb_hash] = [qid for qid, _ in qid_counter.most_common(R)]

        # filter embs and qids
        new_embs, new_qids = [], []
        for emb, qid in zip(embs, qids):
            emb_hash = sha1(emb.tobytes()).hexdigest()
            if qid in emb_qid_d[emb_hash]:
                new_embs.append(emb)
                new_qids.append(qid)

        embs = np.array(new_embs)
        qids = np.array(new_qids)
        return embs, qids
