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
    output_file = os.path.join(output_dir, "embs_qids.npz")
    assert os.path.exists(output_file), "Output file was not created"
    output_data = np.load(output_file)
    return output_data["embs"], output_data["qids"]


def test_output_matches_input(temp_dirs, random_embs_and_qids):
    input_dir, output_dir = temp_dirs
    embs, qids = random_embs_and_qids

    save_embs_and_qids(input_dir, embs, qids)
    combine_embs_by_qid(input_dir, output_dir)
    output_embs, output_qids = load_output(output_dir)

    assert (
        len(output_embs) == len(output_qids) == len(np.unique(qids))
    ), "Number of output embeddings does not match number of unique QIDs"


def test_embeddings_normalized(temp_dirs, random_embs_and_qids):
    input_dir, output_dir = temp_dirs
    embs, qids = random_embs_and_qids

    save_embs_and_qids(input_dir, embs, qids)
    combine_embs_by_qid(input_dir, output_dir)
    output_embs, _ = load_output(output_dir)

    assert np.allclose(
        np.linalg.norm(output_embs, axis=1), 1.0
    ), "Output embeddings are not normalized"


def test_all_qids_present(temp_dirs, random_embs_and_qids):
    input_dir, output_dir = temp_dirs
    embs, qids = random_embs_and_qids

    save_embs_and_qids(input_dir, embs, qids)
    combine_embs_by_qid(input_dir, output_dir)
    _, output_qids = load_output(output_dir)

    assert set(output_qids) == set(
        qids
    ), "Not all original QIDs are present in the output"
