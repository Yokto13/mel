import pytest
from unittest.mock import Mock
from data_processors.tokens.mention_qid_pair import MentionQidPair
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
def qid_parser():
    return lambda x: x


@pytest.fixture
def entry_processor(tokenizer_wrapper, qid_parser):
    return EntryProcessor(tokenizer_wrapper, qid_parser)


@pytest.fixture
def damuel_iterator(entry_processor):
    return DamuelDescriptionsTokensIteratorBoth(
        Path("/path/to/damuel"), entry_processor
    )


def test_qid_parser(qid_parser):
    qid_parser.parse = Mock()
    qid_parser.parse.return_value = 123
    assert qid_parser.parse("qid") == 123


def test_entry_processor(entry_processor):
    entry_processor.tokenizer_wrapper.tokenize = Mock()
    entry_processor.tokenizer_wrapper.tokenize.return_value = {"input_ids": [1, 2, 3]}

    damuel_entry = {"wiki": {"title": "label", "text": "description"}, "qid": 123}
    result = entry_processor.process_both(damuel_entry)

    print(result[0])
    print(result[1])
    print(result[0] == MentionQidPair({"input_ids": [1, 2, 3]}, 123))
    assert result == (
        MentionQidPair({"input_ids": [1, 2, 3]}, 123),
        MentionQidPair({"input_ids": [1, 2, 3]}, 123),
    )


def test_damuel_iterator(damuel_iterator):
    tokenizer_wrapper = Mock()
    tokenizer_wrapper.tokenize = lambda x: x
    entry_processor = EntryProcessor(tokenizer_wrapper, lambda x: int(x[1:]))

    entry_processor.tokenizer_wrapper = tokenizer_wrapper
    damuel_iterator.entry_processor = entry_processor

    lines = [
        '{"wiki": {"title": "label1", "text": "description1"}, "qid": "Q1"}',
        '{"wiki": {"title": "label2", "text": "description2"}, "qid": "Q2"}',
    ]

    results = [r for r in damuel_iterator._iterate_file(lines)]
    assert results == [
        (MentionQidPair("label1", 1), MentionQidPair("description1", 1)),
        (MentionQidPair("label2", 2), MentionQidPair("description2", 2)),
    ]
