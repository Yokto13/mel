import os
import tempfile
from typing import List
import numpy as np
import pytest
from transformers import AutoTokenizer

from tokenization.pipeline.pipelines import MewsliMentionPipeline

THIS_DIR = os.path.dirname(os.path.abspath(__file__))


@pytest.mark.parametrize(
    "expected_size, compress",
    [
        (64, False),
        (16, True),
    ],
)
def test_run_mewsli_mention(expected_size: int, compress: bool) -> None:
    # Use the actual Mewsli file
    mewsli_tsv_path = os.path.join(THIS_DIR, "data", "mentions.tsv")

    # Ensure the file exists
    assert os.path.exists(
        mewsli_tsv_path
    ), f"Mewsli file not found at {mewsli_tsv_path}"

    # Use the same tokenizer as in runner.py
    tokenizer = AutoTokenizer.from_pretrained("setu4993/LEALLA-base")

    # Create a temporary directory for the output
    with tempfile.TemporaryDirectory() as temp_dir:
        # Run the pipeline with the real data
        pipeline = MewsliMentionPipeline(
            mewsli_tsv_path=mewsli_tsv_path,
            tokenizer=tokenizer,
            expected_size=expected_size,
            output_filename=os.path.join(temp_dir, "tokens_qids.npz"),
            compress=compress,
        )
        pipeline.run()

        # Check if the output file was created
        output_file = os.path.join(temp_dir, "tokens_qids.npz")
        assert os.path.exists(output_file)

        # Load the output data and check its contents
        data = np.load(output_file, allow_pickle=True)
        assert "tokens" in data
        assert "qids" in data

        tokens = data["tokens"]
        qids = data["qids"]

        # Check the shape and data type of tokens
        assert (
            tokens.ndim == 2
        ), f"Expected tokens to be 2-dimensional, but got {tokens.ndim} dimensions"
        assert (
            tokens.shape[1] == expected_size
        ), f"Expected tokens to have shape (n, {expected_size}), but got {tokens.shape}"
        assert np.issubdtype(
            tokens.dtype, np.integer
        ), f"Expected tokens to be integer type, but got {tokens.dtype}"

        # Check the shape and data type of qids
        assert (
            qids.ndim == 1
        ), f"Expected qids to be 1-dimensional, but got {qids.ndim} dimensions"
        assert (
            qids.shape[0] == tokens.shape[0]
        ), f"Expected qids to have same length as tokens, but got {qids.shape[0]} vs {tokens.shape[0]}"
        assert np.issubdtype(
            qids.dtype, np.integer
        ), f"Expected qids to be integer type, but got {qids.dtype}"

        # You may want to add more specific checks based on the content of your Mewsli file
        # For example, checking for specific QID values or token patterns you expect to see
