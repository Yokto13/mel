import os
import tempfile
from typing import List
import numpy as np
import pytest
from transformers import AutoTokenizer

from tokenization.pipeline.pipelines import (
    MewsliMentionPipeline,
    MewsliContextPipeline,
    DamuelLinkMentionPipeline,
    DamuelLinkContextPipeline,
    DamuelDescriptionMentionPipeline,
    DamuelDescriptionContextPipeline,
)

THIS_DIR = os.path.dirname(os.path.abspath(__file__))


@pytest.fixture
def standard_tokenizer():
    return AutoTokenizer.from_pretrained("setu4993/LEALLA-base")


@pytest.fixture
def tokenizer_with_label():
    t = AutoTokenizer.from_pretrained("setu4993/LEALLA-base")
    t.add_tokens(["[M]"])
    return t


@pytest.mark.parametrize(
    "expected_size, compress",
    [
        (64, False),
        (16, True),
    ],
)
def test_run_mewsli_mention(
    expected_size: int, compress: bool, standard_tokenizer
) -> None:
    mewsli_tsv_path = os.path.join(THIS_DIR, "data", "mentions.tsv")
    assert os.path.exists(
        mewsli_tsv_path
    ), f"Mewsli file not found at {mewsli_tsv_path}"
    tokenizer = standard_tokenizer

    with tempfile.TemporaryDirectory() as temp_dir:
        pipeline = MewsliMentionPipeline(
            mewsli_tsv_path=mewsli_tsv_path,
            tokenizer=tokenizer,
            expected_size=expected_size,
            output_filename=os.path.join(temp_dir, "tokens_qids.npz"),
            compress=compress,
        )
        pipeline.run()

        output_file = os.path.join(temp_dir, "tokens_qids.npz")
        assert os.path.exists(output_file)

        data = np.load(output_file, allow_pickle=True)
        assert "tokens" in data
        assert "qids" in data

        tokens = data["tokens"]
        qids = data["qids"]

        assert (
            tokens.ndim == 2
        ), f"Expected tokens to be 2-dimensional, but got {tokens.ndim} dimensions"
        assert (
            tokens.shape[1] == expected_size
        ), f"Expected tokens to have shape (n, {expected_size}), but got {tokens.shape}"
        assert np.issubdtype(
            tokens.dtype, np.integer
        ), f"Expected tokens to be integer type, but got {tokens.dtype}"

        assert (
            qids.ndim == 1
        ), f"Expected qids to be 1-dimensional, but got {qids.ndim} dimensions"
        assert (
            qids.shape[0] == tokens.shape[0]
        ), f"Expected qids to have same length as tokens, but got {qids.shape[0]} vs {tokens.shape[0]}"
        assert np.issubdtype(
            qids.dtype, np.integer
        ), f"Expected qids to be integer type, but got {qids.dtype}"


@pytest.mark.parametrize(
    "expected_size, compress",
    [
        (64, False),
        (16, True),
    ],
)
def test_run_mewsli_mention_context(
    expected_size: int, compress: bool, tokenizer_with_label
) -> None:
    label_token = "[M]"
    mewsli_tsv_path = os.path.join(THIS_DIR, "data", "mentions.tsv")
    assert os.path.exists(
        mewsli_tsv_path
    ), f"Mewsli file not found at {mewsli_tsv_path}"
    tokenizer = tokenizer_with_label

    with tempfile.TemporaryDirectory() as temp_dir:
        pipeline = MewsliContextPipeline(
            mewsli_tsv_path=mewsli_tsv_path,
            tokenizer=tokenizer,
            label_token=label_token,
            expected_size=expected_size,
            output_filename=os.path.join(temp_dir, "tokens_qids.npz"),
            compress=compress,
        )
        pipeline.run()

        output_file = os.path.join(temp_dir, "tokens_qids.npz")
        assert os.path.exists(output_file)

        data = np.load(output_file, allow_pickle=True)
        assert "tokens" in data
        assert "qids" in data

        tokens = data["tokens"]
        qids = data["qids"]

        assert (
            tokens.ndim == 2
        ), f"Expected tokens to be 2-dimensional, but got {tokens.ndim} dimensions"
        assert (
            tokens.shape[1] == expected_size
        ), f"Expected tokens to have shape (n, {expected_size}), but got {tokens.shape}"
        assert np.issubdtype(
            tokens.dtype, np.integer
        ), f"Expected tokens to be integer type, but got {tokens.dtype}"

        assert (
            qids.ndim == 1
        ), f"Expected qids to be 1-dimensional, but got {qids.ndim} dimensions"
        assert (
            qids.shape[0] == tokens.shape[0]
        ), f"Expected qids to have same length as tokens, but got {qids.shape[0]} vs {tokens.shape[0]}"
        assert np.issubdtype(
            qids.dtype, np.integer
        ), f"Expected qids to be integer type, but got {qids.dtype}"

        assert np.all(tokens != 0), "Expected no zero tokens in the context"

        label_token_id = tokenizer.encode(label_token, add_special_tokens=False)[0]
        label_token_count = np.sum(tokens == label_token_id, axis=1)
        assert np.all(
            label_token_count == 2
        ), f"Expected {label_token} to appear exactly twice in each row"


@pytest.mark.parametrize(
    "expected_size, compress",
    [
        (64, False),
        (16, True),
    ],
)
def test_run_damuel_link_mention(
    expected_size: int, compress: bool, standard_tokenizer
) -> None:
    damuel_path = os.path.join(THIS_DIR, "data", "damuel")
    assert os.path.exists(damuel_path), f"DaMuEL file not found at {damuel_path}"
    tokenizer = standard_tokenizer

    with tempfile.TemporaryDirectory() as temp_dir:
        pipeline = DamuelLinkMentionPipeline(
            damuel_path=damuel_path,
            tokenizer=tokenizer,
            expected_size=expected_size,
            output_filename=os.path.join(temp_dir, "tokens_qids.npz"),
            compress=compress,
        )
        pipeline.run()

        output_file = os.path.join(temp_dir, "tokens_qids.npz")
        assert os.path.exists(output_file)

        data = np.load(output_file, allow_pickle=True)
        assert "tokens" in data
        assert "qids" in data

        tokens = data["tokens"]
        qids = data["qids"]

        assert (
            tokens.ndim == 2
        ), f"Expected tokens to be 2-dimensional, but got {tokens.ndim} dimensions"
        assert (
            tokens.shape[1] == expected_size
        ), f"Expected tokens to have shape (n, {expected_size}), but got {tokens.shape}"
        assert np.issubdtype(
            tokens.dtype, np.integer
        ), f"Expected tokens to be integer type, but got {tokens.dtype}"

        assert (
            qids.ndim == 1
        ), f"Expected qids to be 1-dimensional, but got {qids.ndim} dimensions"
        assert (
            qids.shape[0] == tokens.shape[0]
        ), f"Expected qids to have same length as tokens, but got {qids.shape[0]} vs {tokens.shape[0]}"
        assert np.issubdtype(
            qids.dtype, np.integer
        ), f"Expected qids to be integer type, but got {qids.dtype}"


@pytest.mark.parametrize(
    "expected_size, compress",
    [
        (64, False),
        (16, True),
    ],
)
def test_run_damuel_link_context(
    expected_size: int, compress: bool, tokenizer_with_label
) -> None:
    label_token = "[M]"
    damuel_path = os.path.join(THIS_DIR, "data", "damuel")
    assert os.path.exists(damuel_path), f"DaMuEL file not found at {damuel_path}"
    tokenizer = tokenizer_with_label

    with tempfile.TemporaryDirectory() as temp_dir:
        pipeline = DamuelLinkContextPipeline(
            damuel_path=damuel_path,
            tokenizer=tokenizer,
            label_token=label_token,
            expected_size=expected_size,
            output_filename=os.path.join(temp_dir, "tokens_qids.npz"),
            compress=compress,
        )
        pipeline.run()

        output_file = os.path.join(temp_dir, "tokens_qids.npz")
        assert os.path.exists(output_file)

        data = np.load(output_file, allow_pickle=True)
        assert "tokens" in data
        assert "qids" in data

        tokens = data["tokens"]
        qids = data["qids"]

        assert (
            tokens.ndim == 2
        ), f"Expected tokens to be 2-dimensional, but got {tokens.ndim} dimensions"
        assert (
            tokens.shape[1] == expected_size
        ), f"Expected tokens to have shape (n, {expected_size}), but got {tokens.shape}"
        assert np.issubdtype(
            tokens.dtype, np.integer
        ), f"Expected tokens to be integer type, but got {tokens.dtype}"

        assert (
            qids.ndim == 1
        ), f"Expected qids to be 1-dimensional, but got {qids.ndim} dimensions"
        assert (
            qids.shape[0] == tokens.shape[0]
        ), f"Expected qids to have same length as tokens, but got {qids.shape[0]} vs {tokens.shape[0]}"
        assert np.issubdtype(
            qids.dtype, np.integer
        ), f"Expected qids to be integer type, but got {qids.dtype}"

        label_token_id = tokenizer.encode(label_token, add_special_tokens=False)[0]
        label_token_count = np.sum(tokens == label_token_id, axis=1)
        assert np.all(
            label_token_count == 2
        ), f"Expected {label_token} to appear exactly twice in each row"


@pytest.mark.parametrize(
    "expected_size, compress",
    [
        (64, False),
        (16, True),
    ],
)
def test_run_damuel_description_mention(
    expected_size: int, compress: bool, standard_tokenizer
) -> None:
    damuel_path = os.path.join(THIS_DIR, "data", "damuel")
    assert os.path.exists(damuel_path), f"DaMuEL file not found at {damuel_path}"
    tokenizer = standard_tokenizer

    with tempfile.TemporaryDirectory() as temp_dir:
        pipeline = DamuelDescriptionMentionPipeline(
            damuel_path=damuel_path,
            tokenizer=tokenizer,
            expected_size=expected_size,
            output_filename=os.path.join(temp_dir, "tokens_qids.npz"),
            compress=compress,
        )
        pipeline.run()

        output_file = os.path.join(temp_dir, "tokens_qids.npz")
        assert os.path.exists(output_file)

        data = np.load(output_file, allow_pickle=True)
        assert "tokens" in data
        assert "qids" in data

        tokens = data["tokens"]
        qids = data["qids"]

        assert (
            tokens.ndim == 2
        ), f"Expected tokens to be 2-dimensional, but got {tokens.ndim} dimensions"
        assert (
            tokens.shape[1] == expected_size
        ), f"Expected tokens to have shape (n, {expected_size}), but got {tokens.shape}"
        assert np.issubdtype(
            tokens.dtype, np.integer
        ), f"Expected tokens to be integer type, but got {tokens.dtype}"

        assert (
            qids.ndim == 1
        ), f"Expected qids to be 1-dimensional, but got {qids.ndim} dimensions"
        assert (
            qids.shape[0] == tokens.shape[0]
        ), f"Expected qids to have same length as tokens, but got {qids.shape[0]} vs {tokens.shape[0]}"
        assert np.issubdtype(
            qids.dtype, np.integer
        ), f"Expected qids to be integer type, but got {qids.dtype}"


@pytest.mark.parametrize(
    "expected_size, compress",
    [
        (64, False),
        (16, True),
    ],
)
def test_run_damuel_description_context(
    expected_size: int, compress: bool, tokenizer_with_label
) -> None:
    label_token = "[M]"
    damuel_path = os.path.join(THIS_DIR, "data", "damuel")
    assert os.path.exists(damuel_path), f"DaMuEL file not found at {damuel_path}"
    tokenizer = tokenizer_with_label

    with tempfile.TemporaryDirectory() as temp_dir:
        pipeline = DamuelDescriptionContextPipeline(
            damuel_path=damuel_path,
            tokenizer=tokenizer,
            label_token=label_token,
            expected_size=expected_size,
            output_filename=os.path.join(temp_dir, "tokens_qids.npz"),
            compress=compress,
        )
        pipeline.run()

        output_file = os.path.join(temp_dir, "tokens_qids.npz")
        assert os.path.exists(output_file)

        data = np.load(output_file, allow_pickle=True)
        assert "tokens" in data
        assert "qids" in data

        tokens = data["tokens"]
        qids = data["qids"]

        assert (
            tokens.ndim == 2
        ), f"Expected tokens to be 2-dimensional, but got {tokens.ndim} dimensions"
        assert (
            tokens.shape[1] == expected_size
        ), f"Expected tokens to have shape (n, {expected_size}), but got {tokens.shape}"
        assert np.issubdtype(
            tokens.dtype, np.integer
        ), f"Expected tokens to be integer type, but got {tokens.dtype}"

        assert (
            qids.ndim == 1
        ), f"Expected qids to be 1-dimensional, but got {qids.ndim} dimensions"
        assert (
            qids.shape[0] == tokens.shape[0]
        ), f"Expected qids to have same length as tokens, but got {qids.shape[0]} vs {tokens.shape[0]}"
        assert np.issubdtype(
            qids.dtype, np.integer
        ), f"Expected qids to be integer type, but got {qids.dtype}"

        label_token_id = tokenizer.encode(label_token, add_special_tokens=False)[0]
        label_token_count = np.sum(tokens == label_token_id, axis=1)
        assert np.all(
            label_token_count == 2
        ), f"Expected {label_token} to appear exactly twice in each row"
