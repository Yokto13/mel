import numpy as np
import pytest
from models.searchers.scann_searcher import ScaNNSearcher
from models.searchers.brute_force_searcher import BruteForceSearcher


@pytest.fixture
def large_random_data():
    np.random.seed(42)
    n_samples = 20000
    n_features = 128
    embs = np.random.rand(n_samples, n_features).astype(np.float32)
    results = np.arange(n_samples)
    return embs, results


@pytest.fixture
def small_random_data():
    np.random.seed(42)
    n_samples = 200
    n_features = 16
    embs = np.random.rand(n_samples, n_features).astype(np.float32)
    results = np.arange(n_samples)
    return embs, results


@pytest.mark.parametrize("searcher_class", [ScaNNSearcher, BruteForceSearcher])
@pytest.mark.slow
def test_searcher_build_large(large_random_data, searcher_class):
    embs, results = large_random_data
    searcher = searcher_class(embs, results)

    try:
        searcher.build()
    except Exception as e:
        pytest.fail(f"{searcher_class.__name__} build failed with error: {e}")


@pytest.mark.parametrize("searcher_class", [ScaNNSearcher, BruteForceSearcher])
@pytest.mark.slow
def test_searcher_find_large(large_random_data, searcher_class):
    embs, results = large_random_data
    searcher = searcher_class(embs, results)

    n_queries = 10
    query_batch = np.random.rand(n_queries, embs.shape[1]).astype(np.float32)

    num_neighbors = 5
    search_results = searcher.find(query_batch, num_neighbors)

    assert search_results.shape == (n_queries, num_neighbors)
    assert np.all(search_results >= 0) and np.all(search_results < len(results))


@pytest.mark.parametrize("searcher_class", [ScaNNSearcher, BruteForceSearcher])
def test_searcher_build_small(small_random_data, searcher_class):
    embs, results = small_random_data
    searcher = searcher_class(embs, results)

    try:
        searcher.build()
    except Exception as e:
        pytest.fail(f"{searcher_class.__name__} build failed with error: {e}")


@pytest.mark.parametrize("searcher_class", [ScaNNSearcher, BruteForceSearcher])
def test_searcher_find_small(small_random_data, searcher_class):
    embs, results = small_random_data
    searcher = searcher_class(embs, results)

    n_queries = 5
    query_batch = np.random.rand(n_queries, embs.shape[1]).astype(np.float32)

    num_neighbors = 3
    search_results = searcher.find(query_batch, num_neighbors)

    assert search_results.shape == (n_queries, num_neighbors)
    assert np.all(search_results >= 0) and np.all(search_results < len(results))


@pytest.mark.parametrize("run_build_from_init", [False, True])
@pytest.mark.slow
def test_scann_search_build_options(large_random_data, run_build_from_init):
    embs, results = large_random_data
    searcher = ScaNNSearcher(embs, results, run_build_from_init=run_build_from_init)

    if run_build_from_init:
        assert hasattr(searcher, "searcher")
    else:
        assert not hasattr(searcher, "searcher")
        searcher.build()
        assert hasattr(searcher, "searcher")
