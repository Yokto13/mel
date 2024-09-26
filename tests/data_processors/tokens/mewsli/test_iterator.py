import pytest
import pandas as pd
from unittest.mock import MagicMock, patch

from data_processors.tokens.mewsli.tokens_iterator import MewsliTokensIterator


@pytest.fixture
def mock_tokenizer():
    m = MagicMock()
    m.return_value.__enter__.return_value = {"input_ids": [1, 2, 3]}
    return m


@pytest.fixture
def mock_data_df():
    return pd.DataFrame(
        {
            "docid": ["doc1", "doc2"],
            "mention": ["mention1", "mention2"],
            "qid": ["Q1", "Q2"],
            "position": [0, 3],
            "length": [2, 3],
        }
    )


@pytest.fixture
def mock_mewsli_tsv_path(tmp_path_factory):
    tsv_file = tmp_path_factory.mktemp("data").joinpath("mewsli.tsv")
    tsv_file.write_text(
        "docid\tmention\tqid\tposition\tlength\n"
        "doc1\tmention1\tQ1\t0\t6\n"
        "doc2\tmention2\tQ2\t5\t5\n"
    )
    text_dir = tsv_file.parent.joinpath("text")
    text_dir.mkdir()
    (text_dir / "doc1").write_text("This is some text for doc1.")
    (text_dir / "doc2").write_text("This is some text for doc2.")
    return tsv_file


@pytest.fixture
def mewsli_tokens_iterator(mock_tokenizer, mock_data_df, mock_mewsli_tsv_path):
    iterator = MewsliTokensIterator(
        mewsli_tsv_path=mock_mewsli_tsv_path,
        tokenizer=mock_tokenizer,
        use_context=False,
        expected_size=64,
    )
    iterator.data_df = mock_data_df
    return iterator


def test_validate_and_get_path(mock_mewsli_tsv_path):
    iterator = MewsliTokensIterator(
        mewsli_tsv_path=mock_mewsli_tsv_path,
        tokenizer=MagicMock(),
        use_context=False,
        expected_size=64,
    )
    assert iterator.mewsli_tsv_path == mock_mewsli_tsv_path


def test_validate_and_get_expected_size_raises(mock_mewsli_tsv_path):
    with pytest.raises(AssertionError):
        MewsliTokensIterator(
            mewsli_tsv_path=mock_mewsli_tsv_path,
            tokenizer=MagicMock(),
            use_context=True,
            expected_size=0,
        )


def test_validate_and_get_expected_size(mock_mewsli_tsv_path):
    iterator = MewsliTokensIterator(
        mewsli_tsv_path=mock_mewsli_tsv_path,
        tokenizer=MagicMock(),
        use_context=True,
        expected_size=12,
    )
    assert iterator.expected_size == 12


def test_iterate_no_context(mock_tokenizer, mock_data_df, mock_mewsli_tsv_path):
    # mock_tokenizer.return_value = "no_context_tokens"
    iterator = MewsliTokensIterator(
        mewsli_tsv_path=mock_mewsli_tsv_path,
        tokenizer=mock_tokenizer,
        use_context=False,
        expected_size=64,
    )
    iterator.data_df = mock_data_df

    result = [p for p in iterator]

    assert len(result) == 2
    assert isinstance(result[0], tuple)
    assert result[0][1] == 1
