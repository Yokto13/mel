from unittest.mock import patch

from finetunings.evaluation.evaluate import evaluate


class TestEvaluate:
    @patch("finetunings.evaluation.evaluate.run_recall_calculation")
    def test_evaluate_default_langs(self, mock_run_recall):
        evaluate("/root", 1)

        expected_damuel_path = "/root/damuel_for_index_2"
        expected_calls = 9  # default 9 languages

        assert mock_run_recall.call_count == expected_calls
        for call in mock_run_recall.call_args_list:
            assert call[0][0] == expected_damuel_path

    @patch("finetunings.evaluation.evaluate.run_recall_calculation")
    def test_evaluate_custom_langs(self, mock_run_recall):
        custom_langs = ["en", "de"]
        evaluate("/root", 2, langs=custom_langs)

        expected_damuel_path = "/root/damuel_for_index_3"
        expected_mewsli_paths = ["/root/mewsli_embs_en_2", "/root/mewsli_embs_de_2"]

        assert mock_run_recall.call_count == 2
        calls = [call[0] for call in mock_run_recall.call_args_list]

        for i, call in enumerate(calls):
            assert call[0] == expected_damuel_path
            assert call[1] == expected_mewsli_paths[i]
