from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from finetunings.finetune_model.data import (
    LightWeightDataset,
    LinksAndDescriptionsTogetherDataset,
)
from finetunings.finetune_model.train import forward_to_embeddings
from finetunings.finetune_model.train_ddp import construct_labels
from transformers import AutoTokenizer
from utils.model_factory import ModelFactory


class TestLinksAndDescriptionsTogetherDataset:
    @pytest.fixture
    def mock_data(self):
        links = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15]])
        descriptions = np.array(
            [
                [101, 102, 103, 104, 105],
                [106, 107, 108, 109, 110],
                [111, 112, 113, 114, 115],
            ]
        )
        Y = np.array([0, 1, 0])
        return links, descriptions, Y

    @pytest.fixture
    def mock_dataset(self, mock_data, tmp_path):
        links, descriptions, Y = mock_data

        # Create a mock NPZ file
        np.savez(tmp_path / "epoch_1.npz", X=links, lines=descriptions, Y=Y)

        with patch(
            "finetunings.finetune_model.data._load_epoch_npz",
            return_value=(links, descriptions, Y),
        ):
            dataset = LinksAndDescriptionsTogetherDataset(tmp_path, 1)
        return dataset

    def test_len(self, mock_dataset):
        assert len(mock_dataset) == 3

    def test_getitem(self, mock_dataset):
        item, label = mock_dataset[1]
        expected_item = np.concatenate(([6, 7, 8, 9, 10], [106, 107, 108, 109, 110]))
        assert np.array_equal(item, expected_item)
        assert label == 1

    def test_links_cnt(self, mock_dataset):
        assert mock_dataset.links_cnt == 5

    def test_descriptions_cnt(self, mock_dataset):
        assert mock_dataset.descriptions_cnt == 5


@pytest.mark.slow
class TestForwardToEmbeddings:
    @pytest.fixture(scope="module")
    def model_and_tokenizer(self):
        # Load the model and tokenizer
        model = ModelFactory.auto_load_from_file(
            "hf-internal-testing/tiny-random-BertModel"
        )
        tokenizer = AutoTokenizer.from_pretrained(
            "hf-internal-testing/tiny-random-BertModel"
        )
        return model, tokenizer

    @pytest.fixture(scope="module")
    def sample_texts(self):
        return [
            "This is a sample sentence.",
            "Another example of input text.",
            "Short text.",
        ]

    def test_forward_to_embeddings(self, model_and_tokenizer, sample_texts):
        model, tokenizer = model_and_tokenizer

        inputs = tokenizer(
            sample_texts, padding=True, truncation=True, return_tensors="pt"
        )
        input_ids = inputs["input_ids"]

        result = forward_to_embeddings(input_ids, model)

        assert result.shape == (len(sample_texts), model.output_dim)

        torch.testing.assert_close(
            torch.norm(result, p=2, dim=1),
            torch.ones(len(sample_texts)),
            atol=1e-4,
            rtol=1e-4,
        )


class TestLightWeightDataset:
    @pytest.fixture
    def mock_data(self):
        links = np.array(
            [[1, 2, 3, 4, 5, 0], [6, 7, 8, 9, 10, 0], [11, 12, 13, 14, 15, 0]]
        )
        descriptions = np.array(
            [
                [101, 102, 103, 104, 105, 0],
                [106, 107, 108, 109, 110, 0],
                [111, 112, 113, 114, 115, 0],
            ]
        )
        return links, descriptions

    def create_mock_dataset(self, mock_data, tmp_path, rank, world_size):
        links, descriptions = mock_data
        np.savez(tmp_path / "epoch_1.npz", X=links, lines=descriptions)
        return LightWeightDataset(tmp_path, 1, rank, world_size)

    @pytest.fixture
    def mock_dataset(self, mock_data, tmp_path):
        return self.create_mock_dataset(mock_data, tmp_path, 0, 1)

    @pytest.fixture
    def mock_dataset_rank_0(self, mock_data, tmp_path):
        return self.create_mock_dataset(mock_data, tmp_path, 0, 3)

    @pytest.fixture
    def mock_dataset_rank_2(self, mock_data, tmp_path):
        return self.create_mock_dataset(mock_data, tmp_path, 2, 3)

    @pytest.fixture
    def mock_dataset_rank_1(self, mock_data, tmp_path):
        return self.create_mock_dataset(mock_data, tmp_path, 1, 3)

    def test_len(
        self,
        mock_dataset,
        mock_dataset_rank_0,
        mock_dataset_rank_1,
        mock_dataset_rank_2,
    ):
        assert len(mock_dataset) == 3
        assert len(mock_dataset_rank_0) == 3
        assert len(mock_dataset_rank_1) == 3
        assert len(mock_dataset_rank_2) == 3

    def test_getitem_basic(self, mock_dataset):
        item = mock_dataset[1]
        expected_item = np.concatenate(
            ([6, 7, 8, 9, 10, 0], [106, 107, 108, 109, 110, 0])
        )
        assert np.array_equal(item, expected_item)

    def test_getitem_rank_0(self, mock_dataset_rank_0):
        item = mock_dataset_rank_0[1]
        expected_item = np.array([6, 7, 8, 9])
        assert np.array_equal(item, expected_item)

        item = mock_dataset_rank_0[0]
        expected_item = np.array([1, 2, 3, 4])
        assert np.array_equal(item, expected_item)

    def test_getitem_rank_1(self, mock_dataset_rank_1):
        item = mock_dataset_rank_1[1]
        expected_item = np.array([10, 0, 106, 107])
        assert np.array_equal(item, expected_item)

    def test_getitem_rank_2(self, mock_dataset_rank_2):
        item = mock_dataset_rank_2[1]
        expected_item = np.array([108, 109, 110, 0])
        assert np.array_equal(item, expected_item)

    def test_links_cnt(
        self,
        mock_dataset,
        mock_dataset_rank_0,
        mock_dataset_rank_1,
        mock_dataset_rank_2,
    ):
        assert mock_dataset.links_cnt == 6
        assert mock_dataset_rank_0.links_cnt == 6
        assert mock_dataset_rank_1.links_cnt == 6
        assert mock_dataset_rank_2.links_cnt == 6

    def test_descriptions_cnt(
        self,
        mock_dataset,
        mock_dataset_rank_0,
        mock_dataset_rank_1,
        mock_dataset_rank_2,
    ):
        assert mock_dataset.descriptions_cnt == 6
        assert mock_dataset_rank_0.descriptions_cnt == 6
        assert mock_dataset_rank_1.descriptions_cnt == 6
        assert mock_dataset_rank_2.descriptions_cnt == 6


class TestDDPUtils:
    def test_construct_labels(self):
        dataset = MagicMock(spec=LightWeightDataset)
        dataset.links_cnt = 3
        dataset.descriptions_cnt = 9

        labels = construct_labels(dataset)

        expected_labels = np.array(
            [
                [1, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0, 0],
            ],
            dtype=np.float32,
        )

        assert np.array_equal(labels, expected_labels)
