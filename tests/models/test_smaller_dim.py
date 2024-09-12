import pytest
import torch
from src.models.smaller_dim import SmallerDim


class MockEmbeddingModel(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)
        self.pooler = torch.nn.Module()
        self.pooler.dense = torch.nn.Linear(output_dim, output_dim)

    def forward(self, tokens, attention_mask):
        x = self.linear(tokens)
        pooler_output = self.pooler.dense(x)
        return type("MockOutput", (), {"pooler_output": pooler_output})()


@pytest.fixture
def mock_embedding_model():
    return MockEmbeddingModel(10, 768)


def test_smaller_dim_forward(mock_embedding_model):
    target_dim = 256
    smaller_dim = SmallerDim(mock_embedding_model, target_dim)

    input_tensor = torch.randn(5, 10)
    attention_mask = torch.ones(5, 10)
    output = smaller_dim(input_tensor, attention_mask)

    assert output.pooler_output.shape == (5, target_dim)


def test_smaller_dim_state_dict(mock_embedding_model):
    target_dim = 128
    original_model = SmallerDim(mock_embedding_model, target_dim)

    state_dict = original_model.state_dict()

    loaded_model = SmallerDim(mock_embedding_model, target_dim)

    loaded_model.load_state_dict(state_dict)

    assert loaded_model.target_dim == target_dim
    assert loaded_model.mapping.out_features == target_dim

    input_tensor = torch.randn(5, 10)
    attention_mask = torch.ones(5, 10)
    with torch.no_grad():
        original_output = original_model(input_tensor, attention_mask)
        loaded_output = loaded_model(input_tensor, attention_mask)

    assert torch.allclose(original_output.pooler_output, loaded_output.pooler_output)


@pytest.mark.parametrize("target_dim", [128, 256, 512])
def test_smaller_dim_with_lealla_base(target_dim):
    from transformers import AutoModel, AutoTokenizer

    # Load the LEALLA-base model and tokenizer
    model_name = "setu4993/LEALLA-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    base_model = AutoModel.from_pretrained(model_name)

    # Initialize SmallerDim with LEALLA-base
    smaller_dim = SmallerDim(base_model, target_dim)

    # Prepare input
    text = ["This is a test sentence.", "Another example input."]
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)

    # Forward pass
    with torch.no_grad():
        output = smaller_dim(inputs.input_ids, inputs.attention_mask)

    # Check output shape
    assert output.pooler_output.shape == (len(text), target_dim)

    # Check that output values are reasonable (not all zeros or infinities)
    assert not torch.isnan(output.pooler_output).any()
    assert not torch.isinf(output.pooler_output).any()
    assert (output.pooler_output != 0).any()
