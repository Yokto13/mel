import numpy as np

from tokenization.pipeline.tokenizers import CuttingTokenizer


class TestCuttingTokenizer:
    def test_context_tokenizer_output_format(self, mocker):
        tokenizer_mock = mocker.Mock()
        tokenizer_mock.tokenize.side_effect = lambda x: np.array([1, 2, 3])

        tokenizer_wrapper_mock = mocker.patch(
            "tokenization.pipeline.tokenizers.base.TokenizerWrapper"
        )
        tokenizer_wrapper_mock.return_value = tokenizer_mock

        tokens_cutter_mock = mocker.patch(
            "tokenization.pipeline.tokenizers.cutting.TokensCutterV3"
        )
        tokens_cutter_mock.return_value.cut_mention_with_context.return_value = (
            np.array([1, 2, 3])
        )

        mention_slices = [slice(0, 5), slice(10, 15)]
        texts = ["Text 1", "Text 2"]
        qids = [1, 2]
        input_gen = zip(mention_slices, texts, qids)

        tokenizer = CuttingTokenizer(
            tokenizer_mock, expected_size=64, label_token="[M]"
        )
        output = list(tokenizer.process(input_gen))

        assert len(output) == len(mention_slices)
        print(output)
        for (tokens, qid), expected_qid in zip(output, qids):
            print(tokens)
            print(qid)
            print(expected_qid)
            assert isinstance(tokens, np.ndarray)
            assert qid == expected_qid
