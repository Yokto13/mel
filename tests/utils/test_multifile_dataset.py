import pytest
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from utils.multifile_dataset import MultiFileDataset


@pytest.fixture
def temp_data_dir(tmp_path):
    for i in range(3):
        data = {"tokens": np.array([1, 2, 3]), "qids": np.array([i, i + 1, i + 2])}
        np.savez(tmp_path / f"test_file_{i}.npz", **data)
    return tmp_path


def test_init(temp_data_dir):
    dataset = MultiFileDataset(temp_data_dir)
    assert dataset.data_dir == temp_data_dir
    assert dataset.file_pattern == "*.npz"
    assert len(dataset.file_list) == 3


def test_get_file_list(temp_data_dir):
    dataset = MultiFileDataset(temp_data_dir)
    file_list = dataset._get_file_list()
    assert len(file_list) == 3
    for file_path in file_list:
        assert file_path.endswith(".npz")
        assert os.path.exists(file_path)


def test_choose_loader(temp_data_dir):
    dataset = MultiFileDataset(temp_data_dir)
    assert dataset._data_loader == dataset._npz_loader

    with pytest.raises(TypeError):
        MultiFileDataset(temp_data_dir, file_pattern="*.txt")


def test_npz_loader(temp_data_dir):
    dataset = MultiFileDataset(temp_data_dir)
    file_path = dataset.file_list[0]
    tokens, qids = dataset._npz_loader(file_path)
    assert isinstance(tokens, np.ndarray)
    assert isinstance(qids, np.ndarray)
    assert tokens.shape == (3,)
    assert qids.shape == (3,)


def test_iter(temp_data_dir):
    dataset = MultiFileDataset(temp_data_dir)
    data_loader = DataLoader(dataset, batch_size=1, num_workers=0)

    all_tokens = []
    all_qids = []
    for tokens, qids in data_loader:
        all_tokens.extend(tokens.numpy().flatten())
        all_qids.extend(qids.numpy().flatten())

    assert len(all_tokens) == 9  # 3 files * 3 tokens each
    assert len(all_qids) == 9  # 3 files * 3 qids each


def test_multi_worker_iter(temp_data_dir):
    dataset = MultiFileDataset(temp_data_dir)
    data_loader = DataLoader(dataset, batch_size=1, num_workers=2)

    all_tokens = []
    all_qids = []
    for tokens, qids in data_loader:
        all_tokens.extend(tokens.numpy().flatten())
        all_qids.extend(qids.numpy().flatten())

    assert len(all_tokens) == 9  # 3 files * 3 tokens each
    assert len(all_qids) == 9  # 3 files * 3 qids each
