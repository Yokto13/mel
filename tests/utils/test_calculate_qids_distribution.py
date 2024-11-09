from pathlib import Path

from pytest import fixture
import numpy as np

from utils.calculate_qids_distribution import calculate_qids_distribution_from_links


@fixture
def data(tmpdir):
    for i in range(10):
        np.savez(tmpdir / f"{i}.npz", qids=np.random.randint(0, 100, size=20))
    return tmpdir


def test_calculate_qids_distribution_from_links(data):
    qids_dist = calculate_qids_distribution_from_links(Path(data), np.arange(100))
    assert qids_dist.shape == (100,)
    assert np.allclose(qids_dist.sum(), 1)
