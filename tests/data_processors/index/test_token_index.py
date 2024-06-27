import numpy as np
import pytest

from data_processors.index.token_index import TokenIndex


@pytest.fixture(scope="module")
def example_data():
    # Generate example data for testing
    embs = np.random.rand(50000, 128)
    qids = np.arange(50000)
    tokens = np.random.randint(0, 100, (50000, 10))
    attentions = np.random.randint(0, 2, (50000, 10))
    return embs, qids, tokens, attentions


def test_token_index_init(example_data):
    embs, qids, tokens, attentions = example_data
    index = TokenIndex(embs, qids, tokens, attentions)
    assert len(index) == len(embs)


def test_token_index_query(example_data):
    embs, qids, tokens, attentions = example_data
    index = TokenIndex(embs, qids, tokens, attentions)
    query_emb = np.random.rand(128)
    query_qid = np.random.choice(qids)
    pos_results, neg_results = index.query(query_emb, query_qid, positive_cnt=1, neg_cnt=7)
    print(pos_results, neg_results)
    assert len(pos_results[0]) == 1  
    assert len(neg_results[0]) == 7  


def test_token_index_query_batched(example_data):
    embs, qids, tokens, attentions = example_data
    index = TokenIndex(embs, qids, tokens, attentions)
    query_embs = np.random.rand(10, 128)
    query_qids = np.random.choice(qids, 10)
    results = index.query_batched(query_embs, query_qids, positive_cnt=1, neg_cnt=7)
    for pos_results, neg_results in results:
        assert len(pos_results[0]) == 1
        assert len(neg_results[0]) == 7


def test_token_index_save_and_load(tmp_path, example_data):
    embs, qids, tokens, attentions = example_data
    index = TokenIndex(embs, qids, tokens, attentions)
    save_path = tmp_path / "test_index"
    index.save(save_path)
    loaded_index = TokenIndex.from_saved(save_path)
    assert len(index) == len(loaded_index)
    assert np.allclose(index.embs, loaded_index.embs)
    assert np.array_equal(index.qids, loaded_index.qids)
    assert np.array_equal(index.tokens, loaded_index.tokens)
    assert np.array_equal(index.attentions, loaded_index.attentions)
