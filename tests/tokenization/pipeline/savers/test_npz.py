import os

import numpy as np
import pytest

from tokenization.pipeline.savers.npz import NPZSaver


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
