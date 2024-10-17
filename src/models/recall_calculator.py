import logging
from collections.abc import Iterable

import numpy as np
import numpy.typing as npt

from models.searchers.searcher import Searcher

_logger = logging.getLogger("models.recall_calculator")


def _get_unique_n(iterable: Iterable, n: int):
    seen = set()
    for i in iterable:
        if i not in seen:
            seen.add(i)
            yield i
        if len(seen) == n:
            break


class RecallCalculator:
    def __init__(self, searcher: Searcher, mode="single") -> None:
        self.searcher = searcher
        self.mode = mode
        if self.mode == "single":
            _logger.info(
                "Initialized RecallCalculator with searcher. "
                "Please note that this RecallCalculator assumes that there is only one embedding per qid in the searcher."
                "If you need to run this with index where there are multiple embeddings per qid (for example Moleman)"
                "you will need to adjust the code that samples from the searcher (retrieve more than k neighbors)"
            )
        elif self.mode == "multiple":
            _logger.info(
                "Initialized RecallCalculator with searcher. "
                "This mode assumes that there are multiple embeddings per qid in the searcher."
                "If this is not the case, consider using the mode='single' instead because this mode is slower."
            )
        else:
            raise ValueError(f"Invalid mode: {self.mode}")

    def recall(self, mewsli_embs, mewsli_qids, k: int):
        qid_was_present = self._process_for_recall(mewsli_embs, mewsli_qids, k)
        return self._calculate_recall(qid_was_present)

    def _calculate_recall(self, qid_was_present):
        return sum(qid_was_present) / len(qid_was_present)

    def _get_neighboring_qids(self, queries_embs, k):
        if self.mode == "single":
            qids_per_query = []
            neighbors_qids = self.searcher.find(queries_embs, k)
            for ns_qids in neighbors_qids:
                unique_ns_qids = list(_get_unique_n(ns_qids, k))
                qids_per_query.append(unique_ns_qids)
            return qids_per_query
        elif self.mode == "multiple":
            retrieval_k = k
            retrieved_enough_neighbors = False
            while not retrieved_enough_neighbors:
                qids_per_query = []
                neighbors_qids = self.searcher.find(queries_embs, retrieval_k)
                for ns_qids in neighbors_qids:
                    unique_ns_qids = list(_get_unique_n(ns_qids, k))
                    qids_per_query.append(unique_ns_qids)

                retrieved_enough_neighbors = True
                for qids in qids_per_query:
                    if len(qids) < k:
                        retrieved_enough_neighbors = False
                        break
                if retrieved_enough_neighbors:
                    break
                retrieval_k *= 2

            return qids_per_query

    def _process_for_recall(self, mewsli_embs, mewsli_qids, k):
        qid_was_present = []

        for emb, qid in zip(mewsli_embs, mewsli_qids):
            # TODO: This should be reworked to batching solution
            negihboring_qids = self._get_neighboring_qids(np.array([emb]), k)
            qid_was_present.append(qid in negihboring_qids[0])

        return qid_was_present
