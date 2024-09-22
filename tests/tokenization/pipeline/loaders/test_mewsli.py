import pytest

from tokenization.pipeline.loaders.mewsli import MewsliLoader


@pytest.fixture
def mewsli_data(tmp_path):
    mentions_file = tmp_path / "mentions.tsv"
    mentions_data = [
        "qid\tdocid\tposition\tlength\tmention",
        "Q1\tdoc1\t0\t5\tmention1",
        "Q2\tdoc2\t10\t8\tmention2",
        "Q3\tdoc3\t20\t7\tmention3",
    ]
    mentions_file.write_text("\n".join(mentions_data))

    text_dir = tmp_path / "text"
    text_dir.mkdir()
    (text_dir / "doc1").write_text("Text for doc1")
    (text_dir / "doc2").write_text("Text for doc2")
    (text_dir / "doc3").write_text("Text for doc3")

    return str(mentions_file), mentions_data


class TestMewsliLoader:
    def test_mewsli_loader_line_count(self, mewsli_data):
        mentions_file, mentions_data = mewsli_data
        loader = MewsliLoader(mentions_file, use_context=True)
        results = list(loader.process())

        assert len(results) == len(mentions_data) - 1

    def test_mewsli_loader_without_context(self, mewsli_data):
        mentions_file, mentions_data = mewsli_data
        loader = MewsliLoader(mentions_file, use_context=False)
        results = list(loader.process())

        assert len(results) == len(mentions_data) - 1
        for (mention, qid), expected in zip(results, mentions_data[1:]):
            expected_qid, _, _, _, expected_mention = expected.split("\t")
            assert mention == expected_mention
