import numpy as np
import pytest

from tokenization.pipeline.filters.base import Filter


def contains_wiki_key(obj: dict) -> bool:
    return "wiki" in obj


def complex_assert(output, expected_output):
    for output_item, expected_output_item in zip(output, expected_output):
        for a, b in zip(output_item, expected_output_item):
            assert a == b


class TestFilter:
    @pytest.mark.parametrize(
        "filter_func, input_data, expected_output",
        [
            (
                contains_wiki_key,
                [{"wiki": "value1"}, {"other": "value2"}, {"wiki": "value3"}],
                [{"wiki": "value1"}, {"wiki": "value3"}],
            ),
            (
                lambda x: x["data"][1] % 2 == 0,
                [
                    {"data": (np.array([1, 2, 3]), 1)},
                    {"data": (np.array([4, 5, 6]), 2)},
                    {"data": (np.array([7, 8, 9]), 3)},
                    {"data": (np.array([10, 11, 12]), 4)},
                ],
                [
                    {"data": (np.array([4, 5, 6]), 2)},
                    {"data": (np.array([10, 11, 12]), 4)},
                ],
            ),
            (
                lambda x: True,
                [{"a": 1}, {"b": 2}, {"c": 3}],
                [{"a": 1}, {"b": 2}, {"c": 3}],
            ),
            (lambda x: False, [{"a": 1}, {"b": 2}, {"c": 3}], []),
        ],
    )
    def test_filter(self, filter_func, input_data, expected_output):
        filter_step = Filter(filter_func)
        output = list(filter_step.process(iter(input_data)))
        complex_assert(output, expected_output)
