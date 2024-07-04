from pytest import fixture
from transformers import BertTokenizerFast

from utils.extractors.extractor_directors import Director
from utils.extractors.extractor_directors import TokenizingParams


@fixture
def name_token():
    return "[M]"


@fixture
def tokenizer(name_token):
    tokenizer = BertTokenizerFast.from_pretrained("setu4993/LEALLA-base")
    tokenizer.add_tokens([name_token])
    return tokenizer


@fixture
def director():
    return Director()


@fixture
def tokenizing_params(tokenizer, name_token):
    return TokenizingParams(tokenizer, 16, name_token)


def test_creates_description_tokens(director: Director, tokenizing_params):
    extractor = director.construct_descriptions_for_finetuning(
        "tests/damuel_mock", tokenizing_params
    )
    print(type(extractor))
    for a in extractor:
        print(a)
        print(tokenizing_params.tokenizer.decode(a[0]))
    assert False
