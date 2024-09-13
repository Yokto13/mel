import pytest
import torch
from transformers import AutoModel, AutoTokenizer

from src.models.pooling_wrappers import (
    CLSWrapper,
    PoolerOutputWrapper,
    SentenceTransformerWrapper,
)


@pytest.fixture(
    params=[
        ("setu4993/LEALLA-small", 128),
        ("sentence-transformers/multi-qa-MiniLM-L6-cos-v1", 384),
    ]
)
def model_setup(request):
    model_path, output_dim = request.param
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path)
    input_ids = tokenizer.encode("Hello, world!", return_tensors="pt")
    attention_mask = torch.ones_like(input_ids)
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

    assert output.shape == (1, output_dim)
    assert torch.is_tensor(output)


def test_cls_wrapper_forward(model_setup):
    model, input_ids, attention_mask, output_dim = model_setup
    cls_wrapper = CLSWrapper(model)

    output = cls_wrapper(input_ids, attention_mask)

    assert output.shape == (1, output_dim)
