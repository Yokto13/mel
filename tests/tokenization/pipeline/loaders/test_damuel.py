import lzma

import pytest

from tokenization.pipeline.loaders import DaMuELStartLoader
from tokenization.pipeline.loaders.damuel import DaMuELDescriptionProcessor


class TestDaMuELStartLoader:
    @pytest.fixture
    def damuel_data(self, tmp_path):
        data_dir = tmp_path / "damuel_data"
        data_dir.mkdir()

        file1 = data_dir / "part-00000"
        file1.write_text('{"id": 1, "text": "Hello"}\n{"id": 2, "text": "World"}')

        file2 = data_dir / "part-00001"
        file2.write_text('{"id": 3, "text": "Foo"}\n{"id": 4, "text": "Bar"}')

        compressed_file = data_dir / "part-00002.xz"
        with lzma.open(compressed_file, "wt") as f:
            f.write('{"id": 5, "text": "Compressed"}\n{"id": 6, "text": "Data"}')

        return str(data_dir)

    def test_damuel_start_loader(self, damuel_data):
        loader = DaMuELStartLoader(damuel_data)
        results = list(loader.process())

        results.sort(key=lambda x: x["id"])

        expected_results = [
            {"id": 1, "text": "Hello"},
            {"id": 2, "text": "World"},
            {"id": 3, "text": "Foo"},
            {"id": 4, "text": "Bar"},
            {"id": 5, "text": "Compressed"},
            {"id": 6, "text": "Data"},
        ]

        assert results == expected_results

    def test_damuel_start_loader_with_remainder_and_mod(self, damuel_data):
        loader = DaMuELStartLoader(damuel_data, remainder=0, mod=2)
        results = list(loader.process())

        results.sort(key=lambda x: x["id"])

        expected_results = [
            {"id": 1, "text": "Hello"},
            {"id": 2, "text": "World"},
            {"id": 5, "text": "Compressed"},
            {"id": 6, "text": "Data"},
        ]

        assert results == expected_results

    def test_damuel_start_loader_with_remainder_and_mod_no_match(self, damuel_data):
        loader = DaMuELStartLoader(damuel_data, remainder=1, mod=2)
        results = list(loader.process())

        results.sort(key=lambda x: x["id"])

        expected_results = [
            {"id": 3, "text": "Foo"},
            {"id": 4, "text": "Bar"},
        ]

        assert results == expected_results


def test_description_title_concatenation_default():
    text = DaMuELDescriptionProcessor.construct_text_from_title_and_description(
        "title", "description"
    )
    assert text == "title\ndescription"


@pytest.mark.parametrize(
    "title, description, original_title, expected_text",
    [
        ("title", "description", "title", "title\ndescription"),
        ("title", "description", "other", "title\ndescription"),
        ("title", "title description", "title", "title\n description"),
        ("title", "description", None, "title\ndescription"),
        ("title", "description", "description", "title\n"),
    ],
)
def test_description_title_concatenation_with_original_title(
    title, description, original_title, expected_text
):
    text = DaMuELDescriptionProcessor.construct_text_from_title_and_description(
        title, description, original_title
    )
    assert text == expected_text
