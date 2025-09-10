import numpy as np
import pytest

pytest.importorskip("faiss")

from models.searchers.brute_force_searcher import BruteForceSearcher


@pytest.fixture
def generate_data():
    def _generate(num_points, dim, num_queries, seed=42):
        rng = np.random.RandomState(seed)
        embs = rng.randn(num_points, dim).astype(np.float32)
        queries = rng.randn(num_queries, dim).astype(np.float32)
        results = np.arange(num_points)
        return embs, queries, results

    return _generate


def assert_equal_results(faiss_results, brute_results):
    assert faiss_results.shape == brute_results.shape
    np.testing.assert_array_equal(faiss_results, brute_results)


def test_small(generate_data):
    from models.searchers.faiss_searcher import FaissSearcher

    embs, queries, results = generate_data(num_points=100, dim=16, num_queries=5)
    bf = BruteForceSearcher(embs, results)
    fs = FaissSearcher(embs, results)
    # fs.build()
    num_neighbors = 3
    brute_out = bf.find(queries, num_neighbors)
    faiss_out = fs.find(queries, num_neighbors)
    assert_equal_results(faiss_out, brute_out)


def test_large(generate_data):
    from models.searchers.faiss_searcher import FaissSearcher

    embs, queries, results = generate_data(
        num_points=10000, dim=64, num_queries=50, seed=123
    )
    bf = BruteForceSearcher(embs, results)
    fs = FaissSearcher(embs, results)
    # fs.build()
    num_neighbors = 10
    brute_out = bf.find(queries, num_neighbors)
    faiss_out = fs.find(queries, num_neighbors)
    assert_equal_results(faiss_out, brute_out)
