import pytest
from unittest.mock import Mock
from pathlib import Path

from transformers import BertTokenizer

from data_processors.tokens.damuel.descriptions.for_finetuning import (
    TokenizerWrapper,
    DamuelDescriptionsTokensIteratorFinetuning,
)


@pytest.fixture
def tokenizer():
    tokenizer = BertTokenizer.from_pretrained("setu4993/LEALLA-small")
    tokenizer.add_tokens(["[M]"])
    return tokenizer


@pytest.fixture
def tokenizer_wrapper(tokenizer):
    return TokenizerWrapper(tokenizer, 64)


@pytest.fixture
def finetuning_damuel_iterator(tokenizer):
    return DamuelDescriptionsTokensIteratorFinetuning(
        Path("/path/to/damuel"), tokenizer, expected_size=64, name_token="[M]"
    )


def test_finetuning_iterator_initialization(finetuning_damuel_iterator):
    assert finetuning_damuel_iterator.expected_size == 64
    assert finetuning_damuel_iterator.name_token == "[M]"


def test_finetuning_iterator_process_to_one(finetuning_damuel_iterator):
    finetuning_damuel_iterator.entry_processor.tokenizer_wrapper.tokenize = Mock()
    finetuning_damuel_iterator.entry_processor.tokenizer_wrapper.tokenize.return_value = [
        1,
        2,
        3,
    ]

    damuel_entry = {"wiki": {"title": "label", "text": "description"}, "qid": "Q1"}
    result = finetuning_damuel_iterator.entry_processor.process_to_one(
        damuel_entry, finetuning_damuel_iterator.name_token
    )

    assert result == ([1, 2, 3], 1)
    assert (
        finetuning_damuel_iterator.entry_processor.tokenizer_wrapper.tokenize.call_args[
            0
        ][0]
        == "[M]label[M] description"
    )


def test_finetuning_iterator_iterate_file(finetuning_damuel_iterator):
    lines = [
        '{"wiki": {"title": "label1", "text": "description1"}, "qid": "Q1"}',
        '{"wiki": {"title": "label2", "text": "description2"}, "qid": "Q2"}',
    ]

    results = [r for r in finetuning_damuel_iterator._iterate_file(lines)]
    assert len(results) == 2
