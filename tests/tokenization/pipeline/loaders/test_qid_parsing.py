from tokenization.pipeline.loaders.qid_parsing import parse_qid

import pytest


@pytest.mark.parametrize(
    "qid, expected",
    [
        ("Q1", 1),
        ("Q42", 42),
        ("Q1000", 1000),
        ("Q9999999", 9999999),
    ],
)
def test_parse_qid_valid(qid: str, expected: int) -> None:
    assert parse_qid(qid) == expected
