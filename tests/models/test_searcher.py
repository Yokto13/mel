import pytest
import numpy as np
from models.searchers.scann_searcher import ScaNNSearcher


@pytest.fixture
def large_random_data():
    np.random.seed(42)
    n_samples = 2000
    n_features = 128
    embs = np.random.rand(n_samples, n_features).astype(np.float32)
    results = np.arange(n_samples)
    return embs, results


def test_scann_searcher_build_large(large_random_data):
    embs, results = large_random_data
    searcher = ScaNNSearcher(embs, results)

    # This test simply checks if the build process completes without errors
    try:
        searcher.build()
    except Exception as e:
        pytest.fail(f"ScaNNSearcher build failed with error: {e}")


def test_scann_searcher_find_large(large_random_data):
    embs, results = large_random_data
    searcher = ScaNNSearcher(embs, results)

    # Generate a small batch of random query vectors
    n_queries = 10
    query_batch = np.random.rand(n_queries, embs.shape[1]).astype(np.float32)

    num_neighbors = 5
    search_results = searcher.find(query_batch, num_neighbors)

    # Basic checks on the search results
    assert search_results.shape == (n_queries, num_neighbors)
    assert np.all(search_results >= 0) and np.all(search_results < len(results))
