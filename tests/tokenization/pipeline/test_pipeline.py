import pytest
import os
import numpy as np
import pandas as pd
from unittest.mock import patch, mock_open
from tokenization.pipeline.pipeline import (
    TokenizationStep,
    TokenizationPipeline,
    DaMuELLoader,
    Filter,
    DaMuELLinkProcessor,
    DaMuELDescriptionProcessor,
    MewsliLoader,
    ContextTokenizer,
    MentionOnlyTokenizer,
    NPZSaver,
    contains_wiki_key,
)
import lzma  # Added import for lzma


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


class TestTokenizationPipeline:
    def test_add_step(self):
        pipeline = TokenizationPipeline()
        step1 = MentionOnlyTokenizer(None, 64)
        step2 = NPZSaver("test.npz")
        pipeline.add(step1)
        pipeline.add(step2)
        assert len(pipeline.steps) == 2
        assert pipeline.steps[0] == step1
        assert pipeline.steps[1] == step2

    def test_run_pipeline(self, mocker):
        pipeline = TokenizationPipeline()
        step1 = mocker.Mock(spec=TokenizationStep)
        step1.process.return_value = ["step1_output1", "step1_output2"]
        step2 = mocker.Mock(spec=TokenizationStep)
        step2.process.return_value = ["step2_output1", "step2_output2"]
        pipeline.add(step1)
        pipeline.add(step2)

        pipeline.run()

        step1.process.assert_called_once_with()
        step2.process.assert_called_once_with(["step1_output1", "step1_output2"])

    def test_pipeline_str_representation(self):
        pipeline = TokenizationPipeline()
        step1 = MentionOnlyTokenizer(None, 64)
        step2 = NPZSaver("test.npz")
        pipeline.add(step1)
        pipeline.add(step2)

        expected_str = (
            "Tokenization Pipeline Steps:\n" "1. MentionOnlyTokenizer\n" "2. NPZSaver"
        )
        assert str(pipeline) == expected_str


class TestDaMuELLoader:
    @pytest.fixture
    def damuel_data(self, tmp_path):
        data_dir = tmp_path / "damuel_data"
        data_dir.mkdir()

        file1 = data_dir / "part-00000"
        file1.write_text('{"id": 1, "text": "Hello"}\n{"id": 2, "text": "World"}')

        file2 = data_dir / "part-00001"
        file2.write_text('{"id": 3, "text": "Foo"}\n{"id": 4, "text": "Bar"}')

        compressed_file = data_dir / "part-00002.xz"
        with lzma.open(compressed_file, "wt") as f:
            f.write('{"id": 5, "text": "Compressed"}\n{"id": 6, "text": "Data"}')

        return str(data_dir)

    def test_damuel_loader(self, damuel_data):
        loader = DaMuELLoader(damuel_data)
        results = list(loader.process())

        results.sort(key=lambda x: x["id"])

        expected_results = [
            {"id": 1, "text": "Hello"},
            {"id": 2, "text": "World"},
            {"id": 3, "text": "Foo"},
            {"id": 4, "text": "Bar"},
            {"id": 5, "text": "Compressed"},
            {"id": 6, "text": "Data"},
        ]

        assert results == expected_results

    def test_damuel_loader_with_remainder_and_mod(self, damuel_data):
        loader = DaMuELLoader(damuel_data, remainder=0, mod=2)
        results = list(loader.process())

        results.sort(key=lambda x: x["id"])

        expected_results = [
            {"id": 1, "text": "Hello"},
            {"id": 2, "text": "World"},
            {"id": 5, "text": "Compressed"},
            {"id": 6, "text": "Data"},
        ]

        assert results == expected_results

    def test_damuel_loader_with_remainder_and_mod_no_match(self, damuel_data):
        loader = DaMuELLoader(damuel_data, remainder=1, mod=2)
        results = list(loader.process())

        results.sort(key=lambda x: x["id"])

        expected_results = [
            {"id": 3, "text": "Foo"},
            {"id": 4, "text": "Bar"},
        ]

        assert results == expected_results


class TestFilter:
    @pytest.mark.parametrize(
        "filter_func, input_data, expected_output",
        [
            (
                contains_wiki_key,
                [{"wiki": "value1"}, {"other": "value2"}, {"wiki": "value3"}],
                [{"wiki": "value1"}, {"wiki": "value3"}],
            ),
            (
                lambda x: x[1] % 2 == 0,
                [
                    (np.array([1, 2, 3]), 1),
                    (np.array([4, 5, 6]), 2),
                    (np.array([7, 8, 9]), 3),
                    (np.array([10, 11, 12]), 4),
                ],
                [(np.array([4, 5, 6]), 2), (np.array([10, 11, 12]), 4)],
            ),
            (
                lambda x: True,
                [{"a": 1}, {"b": 2}, {"c": 3}],
                [{"a": 1}, {"b": 2}, {"c": 3}],
            ),
            (lambda x: False, [{"a": 1}, {"b": 2}, {"c": 3}], []),
        ],
    )
    def test_filter(self, filter_func, input_data, expected_output):
        filter_step = Filter(filter_func)
        output = list(filter_step.process(iter(input_data)))
        assert len(output) == len(expected_output)
        for output_item, expected_item in zip(output, expected_output):
            assert len(output_item) == len(expected_item)
            for a, b in zip(output_item, expected_item):
                np.testing.assert_equal(a, b)


class TestDaMuELLinkProcessor:
    def test_link_processor_output_format(self):
        input_data = [
            {"id": 1, "text": "Text 1", "links": ["link1", "link2"]},
            {"id": 2, "text": "Text 2", "links": ["link3"]},
        ]
        expected_output = [
            "DaMuEL links processed: {'id': 1, 'text': 'Text 1', 'links': ['link1', 'link2']}",
            "DaMuEL links processed: {'id': 2, 'text': 'Text 2', 'links': ['link3']}",
        ]

        processor = DaMuELLinkProcessor()
        output = list(processor.process(iter(input_data)))

        assert output == expected_output


class TestDaMuELDescriptionProcessor:
    def test_description_processor_output_format(self):
        input_data = [
            {"id": 1, "text": "Text 1", "description": "Description 1"},
            {"id": 2, "text": "Text 2", "description": "Description 2"},
        ]
        expected_output = [
            "DaMuEL descriptions processed: {'id': 1, 'text': 'Text 1', 'description': 'Description 1'}",
            "DaMuEL descriptions processed: {'id': 2, 'text': 'Text 2', 'description': 'Description 2'}",
        ]

        processor = DaMuELDescriptionProcessor()
        output = list(processor.process(iter(input_data)))

        assert output == expected_output


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
            assert qid == int(expected_qid[1:])

    def test_parse_qid(self, mewsli_data):
        mentions_file, mentions_data = mewsli_data
        loader = MewsliLoader(mentions_file)
        assert loader._parse_qid("Q123") == 123
        assert loader._parse_qid("Q456789") == 456789


class TestContextTokenizer:
    def test_context_tokenizer_output_format(self, mocker):
        tokenizer_mock = mocker.Mock()
        tokenizer_mock.tokenize.side_effect = lambda x: np.array([1, 2, 3])

        tokenizer_wrapper_mock = mocker.patch(
            "tokenization.pipeline.pipeline.TokenizerWrapper"
        )
        tokenizer_wrapper_mock.return_value = tokenizer_mock

        tokens_cutter_mock = mocker.patch("tokenization.pipeline.pipeline.TokensCutter")
        tokens_cutter_mock.return_value.cut_mention_with_context.return_value = (
            np.array([1, 2, 3])
        )

        mention_slices = [slice(0, 5), slice(10, 15)]
        texts = ["Text 1", "Text 2"]
        qids = [1, 2]
        input_gen = zip(mention_slices, texts, qids)

        tokenizer = ContextTokenizer(tokenizer_mock, expected_size=64)
        output = list(tokenizer.process(input_gen))

        assert len(output) == len(mention_slices)
        print(output)
        for (tokens, qid), expected_qid in zip(output, qids):
            print(tokens)
            print(qid)
            print(expected_qid)
            assert isinstance(tokens, np.ndarray)
            assert qid == expected_qid


class TestMentionOnlyTokenizer:
    def test_mention_only_tokenizer_output_format(self, mocker):
        tokenizer_mock = mocker.Mock()
        tokenizer_mock.tokenize.side_effect = lambda x: np.array([1, 2, 3])

        tokenizer_wrapper_mock = mocker.patch(
            "tokenization.pipeline.pipeline.TokenizerWrapper"
        )
        tokenizer_wrapper_mock.return_value = tokenizer_mock

        mentions = ["mention1", "mention2", "mention3"]
        qids = [1, 2, 3]
        input_gen = zip(mentions, qids)

        tokenizer = MentionOnlyTokenizer(tokenizer_mock, expected_size=64)
        output = list(tokenizer.process(input_gen))

        assert len(output) == len(mentions)
        for (tokens, qid), expected_qid in zip(output, qids):
            assert isinstance(tokens, np.ndarray)
            assert qid == expected_qid


class TestNPZSaver:
    @pytest.fixture
    def saver_data(self):
        tokens = [np.array([1, 2, 3]), np.array([4, 5, 6])]
        qids = [1, 2]
        return tokens, qids

    def test_npz_saver_uncompressed(self, tmp_path, saver_data):
        filename = str(tmp_path / "test_uncompressed.npz")
        tokens, qids = saver_data
        input_gen = zip(tokens, qids)

        saver = NPZSaver(filename)
        list(saver.process(input_gen))

        assert os.path.exists(filename)
        loaded_data = np.load(filename)
        assert "tokens" in loaded_data
        assert "qids" in loaded_data
        np.testing.assert_array_equal(loaded_data["tokens"], np.array(tokens))
        np.testing.assert_array_equal(loaded_data["qids"], np.array(qids))

    def test_npz_saver_compressed(self, tmp_path, saver_data):
        filename = str(tmp_path / "test_compressed.npz")
        tokens, qids = saver_data
        input_gen = zip(tokens, qids)

        saver = NPZSaver(filename, compress=True)
        list(saver.process(input_gen))

        assert os.path.exists(filename)
        loaded_data = np.load(filename)
        assert "tokens" in loaded_data
        assert "qids" in loaded_data
        np.testing.assert_array_equal(loaded_data["tokens"], np.array(tokens))
        np.testing.assert_array_equal(loaded_data["qids"], np.array(qids))

    def test_npz_saver_filename(self, tmp_path, saver_data):
        expected_filename = str(tmp_path / "test_filename.npz")
        tokens, qids = saver_data
        input_gen = zip(tokens, qids)

        saver = NPZSaver(expected_filename)
        list(saver.process(input_gen))

        assert os.path.exists(expected_filename)
