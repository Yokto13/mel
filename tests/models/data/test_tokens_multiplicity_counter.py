import pytest
import numpy as np
from collections import defaultdict
from models.data.only_once_dataset import OnlyOnceTokens
from models.data.tokens_multiplicity_counter import TokensMultiplicityCounter


@pytest.fixture
def tokens_counter():
    return TokensMultiplicityCounter()


def test_initialization(tokens_counter):
    assert isinstance(tokens_counter, OnlyOnceTokens)
    assert isinstance(tokens_counter.memory, defaultdict)


def test_process_hash_new_token(tokens_counter):
    h = 1
    toks = np.array([1, 2, 3])
    count = tokens_counter.process_hash(h, toks)
    assert count == 1
    assert tokens_counter.toks_in_memory(h, toks)


def test_process_hash_existing_token(tokens_counter):
    h = 1
    toks = np.array([1, 2, 3])
    tokens_counter.process_hash(h, toks)
    count = tokens_counter.process_hash(h, toks)
    assert count == 2


def test_add(tokens_counter):
    h = 1
    toks = np.array([1, 2, 3])
    tokens_counter.add(h, toks)
    assert tokens_counter.toks_in_memory(h, toks)
    assert tokens_counter._count(h, toks) == 0


def test_toks_in_memory(tokens_counter):
    h = 1
    toks1 = np.array([1, 2, 3])
    toks2 = np.array([4, 5, 6])
    tokens_counter.add(h, toks1)
    assert tokens_counter.toks_in_memory(h, toks1)
    assert not tokens_counter.toks_in_memory(h, toks2)


def test_increase(tokens_counter):
    h = 1
    toks = np.array([1, 2, 3])
    tokens_counter.add(h, toks)
    tokens_counter.increase(h, toks)
    tokens_counter.increase(h, toks)
    tokens_counter.increase(h, toks)
    assert tokens_counter._count(h, toks) == 3


def test_get_index_in_cell_h_single_token(tokens_counter):
    h = 1
    toks = np.array([1, 2, 3])
    tokens_counter.add(h, toks)
    assert tokens_counter._get_index_in_cell_h(h, toks) == 0


def test_get_index_in_cell_h_multiple_tokens(tokens_counter):
    h = 1
    toks1 = np.array([1, 2, 3])
    toks2 = np.array([4, 5, 6])
    tokens_counter.add(h, toks1)
    tokens_counter.add(h, toks2)
    assert tokens_counter._get_index_in_cell_h(h, toks2) == 1


def test_is_missing_overridden(tokens_counter):
    h = 1
    toks = np.array([1, 2, 3])
    assert tokens_counter.is_missing(h, toks)
    tokens_counter.add(h, toks)
    assert not tokens_counter.is_missing(h, toks)


def test_real_usecase(tokens_counter):
    h = 1
    toks1 = np.array([1, 2, 3])
    toks2 = np.array([4, 5, 6])
    count1 = tokens_counter(toks1)
    count1 = tokens_counter(toks1)

    assert count1 == 2

    count2 = tokens_counter(toks2)
    count2 = tokens_counter(toks2)

    assert count2 == 2

    count1 = tokens_counter(toks1)

    assert count1 == 3
