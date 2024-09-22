import numpy as np

from tokenization.pipeline.tokenizers import SimpleTokenizer


class TestSimpleTokenizer:
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

        tokenizer = SimpleTokenizer(tokenizer_mock, expected_size=64)
        output = list(tokenizer.run(input_gen))

        assert len(output) == len(mentions)
        for (tokens, qid), expected_qid in zip(output, qids):
            assert isinstance(tokens, np.ndarray)
            assert qid == expected_qid
