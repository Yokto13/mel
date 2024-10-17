from unittest.mock import MagicMock

import numpy as np
import pytest

from models.recall_calculator import _get_unique_n, RecallCalculator
from models.searchers.scann_searcher import ScaNNSearcher


class Searcher:
    def find(self, queries_embs, num_neighbors):
        pass


@pytest.fixture
def mock_searcher():
    return MagicMock(spec=Searcher)


@pytest.fixture
def recall_calculator(mock_searcher):
    return RecallCalculator(searcher=mock_searcher)


def test_calculate_recall(recall_calculator):
    qid_was_present = [True, False, True]
    recall = recall_calculator._calculate_recall(qid_was_present)
    assert recall == 2 / 3


def test_calculate_real():
    damuel_embs = np.random.random((50000, 128))
    mewsli_embs = np.random.random((1000, 128))

    damuel_qids = np.random.randint(1, 10000, size=50000)
    mewsli_qids = np.random.randint(1, 10000, size=1000)

    searcher = ScaNNSearcher(damuel_embs, damuel_qids)

    rc = RecallCalculator(searcher)

    recall = rc.recall(mewsli_embs, mewsli_qids, 10)

    assert 0.0 <= recall <= 1.0


def test_get_unique_n_basic():
    iterable = [1, 2, 3, 4, 5]
    n = 3
    result = list(_get_unique_n(iterable, n))
    assert result == [1, 2, 3]


def test_get_unique_n_exact_n():
    iterable = [1, 2, 3]
    n = 3
    result = list(_get_unique_n(iterable, n))
    assert result == [1, 2, 3]


def test_get_unique_n_more_n_than_unique():
    iterable = [1, 2, 2, 3, 3, 3]
    n = 5
    result = list(_get_unique_n(iterable, n))
    assert result == [1, 2, 3]
