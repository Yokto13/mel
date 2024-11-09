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


def test_calculate_qids_distribution_from_links_with_transform(data):
    qids_dist1 = calculate_qids_distribution_from_links(Path(data), np.arange(100))
    qids_dist2 = calculate_qids_distribution_from_links(
        Path(data), np.arange(100), lambda x: x**2
    )
    qids_dist3 = calculate_qids_distribution_from_links(
        Path(data), np.arange(100), lambda x: x**3
    )
    qids_dist4 = calculate_qids_distribution_from_links(
        Path(data), np.arange(100), lambda x: x**x
    )

    for dist in [qids_dist1, qids_dist2, qids_dist3, qids_dist4]:
        assert np.allclose(dist.sum(), 1)
        assert dist.shape == (100,)

    assert max(qids_dist1) < max(qids_dist2) < max(qids_dist3) < max(qids_dist4)
