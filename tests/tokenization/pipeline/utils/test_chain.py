import lzma

import pytest

from tokenization.pipeline.loaders import DaMuELStartLoader
from tokenization.pipeline.utils.chain import ChainStep


class TestChainStep:
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

    def test_chain_three(self, damuel_data):
        loader = DaMuELStartLoader(damuel_data)
        results = list(loader.process())

        expected_results = results + results + results

        chain = ChainStep([loader, loader, loader])
        results = list(chain.process())

        assert results == expected_results

    def test_chain_one(self, damuel_data):
        loader = DaMuELStartLoader(damuel_data)
        results = list(loader.process())

        expected_results = results

        chain = ChainStep([loader])
        results = list(chain.process())

        assert results == expected_results

    def test_raises_value_error(self, damuel_data):
        loader = DaMuELStartLoader(damuel_data)

        chain = ChainStep([loader])

        with pytest.raises(ValueError):
            list(chain.process(1))
