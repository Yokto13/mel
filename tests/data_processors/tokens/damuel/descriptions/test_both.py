import pytest
from unittest.mock import Mock
from pathlib import Path

from transformers import BertTokenizer

from data_processors.tokens.damuel.descriptions.both import (
    TokenizerWrapper,
    EntryProcessor,
    DamuelDescriptionsTokensIteratorBoth,
)


@pytest.fixture
def tokenizer():
    return BertTokenizer.from_pretrained("setu4993/LEALLA-small")


@pytest.fixture
def tokenizer_wrapper(tokenizer):
    return TokenizerWrapper(tokenizer, 64)


@pytest.fixture
def entry_processor(tokenizer_wrapper):
    return EntryProcessor(tokenizer_wrapper)


@pytest.fixture
def damuel_iterator(entry_processor):
    return DamuelDescriptionsTokensIteratorBoth(
        Path("/path/to/damuel"), entry_processor
    )


def test_entry_processor(entry_processor):
    entry_processor.tokenizer_wrapper.tokenize = Mock()
    entry_processor.tokenizer_wrapper.tokenize.return_value = [1, 2, 3]

    damuel_entry = {"wiki": {"title": "label", "text": "description"}, "qid": "Q123"}
    result = entry_processor.process_both(damuel_entry)

    print(result[0])
    print(result[1])
    print(result[0] == ([1, 2, 3], 123))
    assert result == (
        ([1, 2, 3], 123),
        ([1, 2, 3], 123),
    )


def test_damuel_iterator(damuel_iterator):
    tokenizer_wrapper = Mock()
    tokenizer_wrapper.tokenize = lambda x: x
    entry_processor = EntryProcessor(tokenizer_wrapper)

    entry_processor.tokenizer_wrapper = tokenizer_wrapper
    damuel_iterator.entry_processor = entry_processor

    lines = [
        '{"wiki": {"title": "label1", "text": "description1"}, "qid": "Q1"}',
        '{"wiki": {"title": "label2", "text": "description2"}, "qid": "Q2"}',
    ]

    results = [r for r in damuel_iterator._iterate_file(lines)]
    assert results == [
        (("label1", 1), ("description1", 1)),
        (("label2", 2), ("description2", 2)),
    ]
