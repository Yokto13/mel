import numpy as np
import torch

from src.reranking.training.reranking_iterable_dataset import RerankingIterableDataset


def test_reranking_iterable_dataset_iterates_samples(tmp_path):
    file_path = tmp_path / "part-000.npz"
    data = {
        "qids": np.array([10, 20], dtype=np.int64),
        "y": np.array([1.0, 0.0], dtype=np.float32),
        "description_tokens": np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int64),
        "link_tokens": np.array([[7, 8], [9, 10]], dtype=np.int64),
    }
    np.savez(file_path, **data)

    dataset = RerankingIterableDataset(tmp_path)

    samples = list(dataset)
    assert len(samples) == 2

    first = samples[0]
    assert isinstance(first, tuple)
    assert torch.equal(first[2], torch.tensor(1.0, dtype=torch.float32))
