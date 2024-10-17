import numpy as np
import pytest
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from utils.embeddings import embed
from utils.model_factory import ModelFactory


class SimpleDataset(Dataset):
    def __init__(self, texts, qids):
        self.texts = texts
        self.qids = qids
        self.tokenizer = AutoTokenizer.from_pretrained(
            "hf-internal-testing/tiny-random-BertModel"
        )

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        qid = self.qids[idx]
        tokens = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=128,
            return_tensors="pt",
        )
        return tokens["input_ids"].squeeze(), qid


@pytest.fixture
def model():
    return ModelFactory.auto_load_from_file("hf-internal-testing/tiny-random-BertModel")


@pytest.fixture
def dataset():
    texts = [
        "\n\nLorem ipsum dolor sit amet, consectetur adipiscing elit",
        " Nullam mattis, odio id varius tempor, quam dui condimentum massa, vel interdum lorem sem in dui",
        " Duis risus eros, lacinia in vulputate et, suscipit eu justo",
        " Aliquam efficitur semper augue, tincidunt vulputate quam porttitor sed",
        " Sed rutrum accumsan euismod",
        " Nulla eu volutpat purus",
        " Donec pellentesque erat eget metus molestie, nec blandit mauris laoreet",
        " Sed sed mollis justo, nec semper leo",
        " Praesent sapien augue, gravida ac felis iaculis, dignissim tristique nisi",
        " Suspendisse potenti",
        " Praesent nec risus nisi",
        " Sed congue commodo sem, eu suscipit ex facilisis et",
        " Vestibulum tortor eros, bibendum id malesuada quis, auctor nec nulla",
        " Nulla viverra lacinia tortor, id faucibus purus elementum in",
        " Suspendisse aliquet ullamcorper faucibus",
        " Quisque porttitor eros nec egestas elementum",
        " Aliquam id arcu ac turpis posuere vestibulum",
        "\n\nNunc interdum, velit eu tristique ultricies, sapien sapien pulvinar mauris, in mollis urna enim sed ex",
        " Fusce laoreet turpis diam, a laoreet eros vehicula vel",
        " Integer id luctus orci",
        " Quisque vestibulum risus nibh, nec pharetra augue viverra a",
        " Phasellus dignissim tellus vitae vulputate dictum",
        " Praesent eu tincidunt metus, nec venenatis est",
        " Nunc varius interdum sapien, in tincidunt diam",
        " Suspendisse id semper risus",
        " Mauris luctus, libero vel ornare vehicula, risus urna accumsan arcu, eu tristique eros eros in turpis",
        " Aenean gravida feugiat ligula quis scelerisque",
        " Orci varius natoque penatibus et magnis dis parturient montes, nascetur ridiculus mus",
        " Ut ornare purus ac tristique auctor",
        " Pellentesque nec tortor vel urna ultricies hendrerit",
        " Donec quis faucibus diam",
        " Suspendisse orci ipsum, auctor vitae urna at, iaculis iaculis turpis",
        "\n\nCurabitur non tortor quam",
        " Phasellus eget lectus in dolor fermentum gravida",
        " Praesent tristique diam a lorem laoreet, ut faucibus neque aliquam",
        " Cras quis lobortis massa",
        " Nunc a libero quis leo porttitor tempor vitae vitae enim",
        " Nunc velit metus, molestie ac arcu ac, hendrerit volutpat mi",
        " Cras bibendum est nec tortor consequat, vitae viverra risus ornare",
        " Donec pretium neque at feugiat placerat",
        " Nunc at dui gravida elit malesuada tincidunt",
        " Nulla nunc justo, eleifend sit amet lorem sollicitudin, efficitur aliquet dui",
        "\n\nInteger in imperdiet diam, quis euismod libero",
        " Proin condimentum euismod finibus",
        " Aliquam varius pretium arcu eu condimentum",
        " Duis lacus ipsum, luctus vel mauris at, venenatis pulvinar quam",
        " Vestibulum ante ipsum primis in faucibus orci luctus et ultrices posuere cubilia curae; Proin posuere dictum est, non rhoncus lacus aliquam eget",
        " Proin sollicitudin ornare tellus a fermentum",
        " Fusce sed lacus dignissim, lacinia ex in, egestas nulla",
        " Nunc rhoncus dui in risus faucibus, non tincidunt odio molestie",
        " Ut vel sodales nulla, sed efficitur turpis",
        " Sed orci orci, maximus ut nunc vitae, ornare porttitor diam",
        "\n\nMauris eleifend auctor scelerisque",
        " Vivamus fringilla risus ipsum, scelerisque mattis velit blandit eu",
        " Sed et suscipit lorem",
        " Phasellus dapibus finibus erat, in porta massa commodo non",
        " Pellentesque et nisi congue ligula tempor semper sed at felis",
        " Vestibulum porttitor aliquet viverra",
        " Integer ultricies turpis dolor, consectetur sagittis urna laoreet at",
        " Mauris elementum tempor enim, et consectetur nibh accumsan sit amet",
        " Aenean ultricies diam mauris, in tempor dui vulputate sit amet",
        " Integer id commodo erat",
        " Aenean sagittis finibus ante, lacinia consequat tortor",
        " Praesent id massa dolor",
        " Phasellus vitae cursus quam",
    ]
    qids = list(range(len(texts)))
    return SimpleDataset(texts, qids)


def test_embed_basic(model, dataset):
    batch_size = 16
    result = embed(dataset, model, batch_size=batch_size)

    assert len(result) == 2  # embeddings and qids
    embeddings, qids = result

    assert isinstance(embeddings, np.ndarray)
    assert isinstance(qids, np.ndarray)
    assert embeddings.shape[0] == len(dataset)
    assert embeddings.shape[1] == model.model.config.hidden_size
    assert qids.shape[0] == len(dataset)
    assert embeddings.dtype == np.float16


def test_embed_no_qids(model, dataset):
    batch_size = 16
    result = embed(dataset, model, batch_size=batch_size, return_qids=False)

    assert len(result) == 1  # only embeddings
    embeddings = result[0]

    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape[0] == len(dataset)
    assert embeddings.shape[1] == model.model.config.hidden_size


def test_embed_with_tokens(model, dataset):
    batch_size = 16
    result = embed(dataset, model, batch_size=batch_size, return_tokens=True)

    assert len(result) == 3  # embeddings, qids, and tokens
    embeddings, qids, tokens = result

    assert isinstance(embeddings, np.ndarray)
    assert isinstance(qids, np.ndarray)
    assert isinstance(tokens, np.ndarray)
    assert embeddings.shape[0] == len(dataset)
    assert embeddings.shape[1] == model.model.config.hidden_size
    assert qids.shape[0] == len(dataset)
    assert tokens.shape[0] == len(dataset)
    assert tokens.shape[1] == 128  # max_length in the dataset


def test_embed_normalization(model, dataset):
    batch_size = 16
    result = embed(dataset, model, batch_size=batch_size)

    embeddings = result[0]

    # Check if embeddings are normalized (L2 norm should be close to 1)
    norms = np.linalg.norm(embeddings, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-3)
