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
    def __init__(self, searcher: Searcher) -> None:
        self.searcher = searcher
        _logger.info(
            "Initialized RecallCalculator with searcher. "
            "Please note that this RecallCalculator assumes that there is only one embedding per qid in the searcher."
            "If you need to run this with index where there are multiple embeddings per qid (for example Moleman)"
            "you will need to adjust the code that samples from the searcher (retrieve more than k neighbors)"
        )

    def recall(self, mewsli_embs, mewsli_qids, k: int):
        qid_was_present = self._process_for_recall(mewsli_embs, mewsli_qids, k)
        return self._calculate_recall(qid_was_present)

    def _calculate_recall(self, qid_was_present):
        return sum(qid_was_present) / len(qid_was_present)

    def _get_neighboring_qids(self, queries_embs, k):
        qids_per_query = []
        neighbors_qids = self.searcher.find(queries_embs, k)
        for ns_qids in neighbors_qids:
            unique_ns_qids = list(_get_unique_n(ns_qids, k))
            qids_per_query.append(unique_ns_qids)
        return qids_per_query

    def _process_for_recall(self, mewsli_embs, mewsli_qids, k):
        qid_was_present = []

        for emb, qid in zip(mewsli_embs, mewsli_qids):
            # TODO: This should be reworked to batching solution
            negihboring_qids = self._get_neighboring_qids(np.array([emb]), k)
            qid_was_present.append(qid in negihboring_qids[0])

        return qid_was_present
