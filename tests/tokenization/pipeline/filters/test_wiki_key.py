import pytest
from tokenization.pipeline.filters.wiki_key import WikiKeyFilter


class TestWikiKeyFilter:
    @pytest.mark.parametrize(
        "input_data, expected_output",
        [
            (
                [
                    {"wiki": "value1", "key1": "val1", "key2": "val2"},
                    {"other": "value2", "key3": "val3", "key4": "val4"},
                    {"wiki": "value3", "key5": "val5", "key6": "val6"},
                ],
                [
                    {"wiki": "value1", "key1": "val1", "key2": "val2"},
                    {"wiki": "value3", "key5": "val5", "key6": "val6"},
                ],
            ),
            (
                [
                    {
                        "other_wiki": "value1",
                        "key1": "val1",
                        "key2": "val2",
                        "key3": "val3",
                    },
                    {
                        "another": "value2",
                        "key4": "val4",
                        "key5": "val5",
                        "key6": "val6",
                    },
                    {
                        "complex_key_with_wiki_inside": "value3",
                        "key7": "val7",
                        "key8": "val8",
                        "key9": "val9",
                    },
                    {
                        "wiki": "value4",
                        "key10": "val10",
                        "key11": "val11",
                        "key12": "val12",
                    },
                ],
                [
                    {
                        "wiki": "value4",
                        "key10": "val10",
                        "key11": "val11",
                        "key12": "val12",
                    }
                ],
            ),
        ],
    )
    def test_wiki_key_filter(
        self, input_data: list[dict], expected_output: list[dict]
    ) -> None:
        filter_step = WikiKeyFilter()
        output = list(filter_step.process(iter(input_data)))
        assert output == expected_output
