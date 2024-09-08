import pytest
import torch
from src.models.smaller_dim import SmallerDim


class MockEmbeddingModel(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)
        self.pooler = torch.nn.Module()
        self.pooler.dense = torch.nn.Linear(output_dim, output_dim)

    def forward(self, x):
        x = self.linear(x)
        return self.pooler.dense(x)


@pytest.fixture
def mock_embedding_model():
    return MockEmbeddingModel(10, 768)


def test_smaller_dim_forward(mock_embedding_model):
    target_dim = 256
    smaller_dim = SmallerDim(mock_embedding_model, target_dim)

    input_tensor = torch.randn(5, 10)
    output = smaller_dim(input_tensor)

    assert output.shape == (5, target_dim)


def test_smaller_dim_state_dict(mock_embedding_model):
    target_dim = 128
    original_model = SmallerDim(mock_embedding_model, target_dim)

    state_dict = original_model.state_dict()

    loaded_model = SmallerDim(mock_embedding_model, target_dim)

    loaded_model.load_state_dict(state_dict)

    assert loaded_model.target_dim == target_dim
    assert loaded_model.mapping.out_features == target_dim

    input_tensor = torch.randn(5, 10)
    with torch.no_grad():
        original_output = original_model(input_tensor)
        loaded_output = loaded_model(input_tensor)

    assert torch.allclose(original_output, loaded_output)
