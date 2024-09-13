import pytest
from models.change_dim_wrapper import ChangeDimWrapper
from models.pooling_wrappers import PoolerOutputWrapper, SentenceTransformerWrapper
from src.utils.model_builder import ModelBuilder, OutputType

from transformers import AutoTokenizer


@pytest.fixture
def model_builder():
    return ModelBuilder("setu4993/LEALLA-small")


def test_init(model_builder):
    assert model_builder.model_path == "setu4993/LEALLA-small"


def test_set_output_type_cls(model_builder):
    model_builder.set_output_type(OutputType.CLS)
    assert model_builder.output_type == OutputType.CLS

    with pytest.raises(AttributeError):
        model_builder.output_type = OutputType.CLS


def test_set_output_type_sentence_transformer(model_builder):
    model_builder.set_output_type(OutputType.SENTENCE_TRANSFORMER)
    assert model_builder.output_type == OutputType.SENTENCE_TRANSFORMER

    with pytest.raises(AttributeError):
        model_builder.output_type = OutputType.SENTENCE_TRANSFORMER


def test_set_dim(model_builder):
    model_builder.set_dim(768)

    assert model_builder.dim == 768

    with pytest.raises(AttributeError):
        model_builder.dim = 768


@pytest.mark.parametrize(
    "output_type, wrapper_class",
    [
        (OutputType.CLS, PoolerOutputWrapper),
        (OutputType.SENTENCE_TRANSFORMER, SentenceTransformerWrapper),
    ],
)
def test_build_no_set_dim(model_builder, output_type, wrapper_class):
    model_builder.set_output_type(output_type)

    assert isinstance(model_builder.build(), wrapper_class)


@pytest.mark.parametrize(
    "output_type, dim",
    [
        (OutputType.CLS, 512),
        (OutputType.CLS, 64),
        (OutputType.SENTENCE_TRANSFORMER, 512),
        (OutputType.SENTENCE_TRANSFORMER, 64),
    ],
)
def test_build_set_dim(model_builder, output_type, dim):
    model_builder.set_output_type(output_type)
    model_builder.set_dim(dim)

    assert isinstance(model_builder.build(), ChangeDimWrapper)

    tokenizer = AutoTokenizer.from_pretrained(model_builder.model_path)
    tokenized = tokenizer("This is a test sentence.", return_tensors="pt")
    model = model_builder.build()
    output = model(tokenized.input_ids, tokenized.attention_mask)
    assert output.shape == (1, dim)
