import pytest
import torch

from src.models.pooling_wrappers import (
    CLSWrapper,
    PoolerOutputWrapper,
    SentenceTransformerWrapper,
)
from transformers import AutoModel, AutoTokenizer


@pytest.fixture
def model_setup(request):
    model_path = "hf-internal-testing/tiny-random-BertModel"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path)
    input_ids = tokenizer.encode("Hello, world!", return_tensors="pt")
    attention_mask = torch.ones_like(input_ids)
    output_dim = 32
    return model, input_ids, attention_mask, output_dim


def test_cls_wrapper_forward(model_setup):
    model, input_ids, attention_mask, output_dim = model_setup
    cls_wrapper = PoolerOutputWrapper(model)

    output = cls_wrapper(input_ids, attention_mask)

    assert output.shape == (1, output_dim)
    assert torch.is_tensor(output)


def test_sentence_transformer_wrapper_forward(model_setup):
    model, input_ids, attention_mask, output_dim = model_setup
    sentence_transformer_wrapper = SentenceTransformerWrapper(model)

    output = sentence_transformer_wrapper(input_ids, attention_mask)

    assert tuple(output.shape) == (1, output_dim)
    assert torch.is_tensor(output)


def test_cls_wrapper_forward(model_setup):
    model, input_ids, attention_mask, output_dim = model_setup
    cls_wrapper = CLSWrapper(model)

    output = cls_wrapper(input_ids, attention_mask)

    assert output.shape == (1, output_dim)
