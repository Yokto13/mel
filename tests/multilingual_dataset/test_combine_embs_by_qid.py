import pytest
import numpy as np
import os
import tempfile
from src.multilingual_dataset.combine_embs import combine_embs_by_qid


@pytest.fixture
def random_embs_and_qids():
    num_embs = 10000
    emb_dim = 128
    num_unique_qids = 2000

    embs = np.random.rand(num_embs, emb_dim).astype(np.float32)
    qids = np.random.randint(0, num_unique_qids, num_embs)
    return embs, qids


@pytest.fixture
def temp_dirs():
    with tempfile.TemporaryDirectory() as input_dir, tempfile.TemporaryDirectory() as output_dir:
        yield input_dir, output_dir


def save_embs_and_qids(input_dir, embs, qids):
    np.savez_compressed(os.path.join(input_dir, "embs_qids.npz"), embs=embs, qids=qids)


def load_output(output_dir):
    output_file = os.path.join(output_dir, "embs_and_qids.npz")
    assert os.path.exists(output_file), "Output file was not created"
    output_data = np.load(output_file)
    return output_data["embs"], output_data["qids"]


def test_combine_embs_by_qid(temp_dirs, random_embs_and_qids):
    input_dir, output_dir = temp_dirs
    embs, qids = random_embs_and_qids

    save_embs_and_qids(input_dir, embs, qids)
    combine_embs_by_qid(input_dir, output_dir)
    output_embs, output_qids = load_output(output_dir)

    assert_output_matches_input(output_embs, output_qids, qids)
    assert_embeddings_normalized(output_embs)
    assert_all_qids_present(output_qids, qids)


def assert_output_matches_input(output_embs, output_qids, input_qids):
    assert (
        len(output_embs) == len(output_qids) == len(np.unique(input_qids))
    ), "Number of output embeddings does not match number of unique QIDs"


def assert_embeddings_normalized(embs):
    assert np.allclose(
        np.linalg.norm(embs, axis=1), 1.0
    ), "Output embeddings are not normalized"


def assert_all_qids_present(output_qids, input_qids):
    assert set(output_qids) == set(
        input_qids
    ), "Not all original QIDs are present in the output"
