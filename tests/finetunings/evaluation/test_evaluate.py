from unittest import mock

import pytest

from finetunings.evaluation import evaluate


@pytest.fixture
def mock_logger(monkeypatch):
    monkeypatch.setattr(evaluate, "_logger", mock.Mock())


def test_evaluate(monkeypatch, mock_logger):
    dummy_embs = [[1, 2], [3, 4]]
    dummy_qids = [10, 20]
    langs = ["en", "de"]
    calls = []
    monkeypatch.setattr(
        evaluate,
        "load_embs_and_qids_with_normalization",
        lambda path: (dummy_embs, dummy_qids),
    )

    class DummySearcher:
        def __init__(self, embs, qids):
            self.embs = embs
            self.qids = qids

    monkeypatch.setattr(evaluate, "BruteForceSearcher", DummySearcher)

    def fake_find_recall_with_searcher(searcher, mewsli_path, recalls):
        calls.append((searcher, mewsli_path, recalls))

    monkeypatch.setattr(
        evaluate, "find_recall_with_searcher", fake_find_recall_with_searcher
    )
    evaluate.evaluate("/root", 1, langs=langs)
    assert len(calls) == len(langs)
    for i, lang in enumerate(langs):
        assert calls[i][1] == f"/root/mewsli_embs_{lang}_1"
        assert calls[i][2] == [1, 10, 100]
