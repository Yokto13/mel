from pathlib import Path

import numpy as np
import pytest
import torch

from data_processors.index.index import Index 


@pytest.fixture()
def data():
    embs = np.random.rand(100, 128)
    qids = np.random.randint(0, 10, 100)
    return embs, qids


def test_index_init(data):
    embs, qids = data
    index = Index(embs, qids)
    assert len(index) == len(embs)


def test_index_build_index(data):
    embs, qids = data
    index = Index(embs, qids)
    index.build_index()
    assert index.scann_index is not None


def test_index_from_dir(data, tmp_path):
    embs, qids = data
    save_dir = tmp_path / "example_data"
    save_dir.mkdir()
    np.save(save_dir / "embs_123.npy", embs)
    np.save(save_dir / "qids_123.npy", qids)
    index = Index.from_dir(save_dir)
    assert len(index) == len(embs)


def test_index_from_iterable_and_model(data):
    embs, qids = data
    class ExampleModel:
        def forward_only_embeddings(self, embs):
            return embs

        def to(self, device):
            pass

    model = ExampleModel()
    dataloader = torch.utils.data.DataLoader(
        list(zip(embs, qids)), batch_size=1
    )
    index = Index.from_iterable_and_model(dataloader, model)
    assert len(index) == len(embs)


def test_is_emb_file():
    assert Index.is_emb_file(Path("embs_123.npy")) is True
    assert Index.is_emb_file(Path("qids_123.npy")) is False


def test_extract_hash():
    assert Index.extract_hash(Path("embs_123.npy")) == "123"
    assert Index.extract_hash(Path("qids_123.npy")) == "123"
