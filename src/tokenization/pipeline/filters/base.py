from collections.abc import Generator
from typing import Any, Callable

from ..base import PipelineStep


class Filter(PipelineStep):
    def __init__(self, filter_func: Callable[[dict], bool]):
        self.filter_func = filter_func

    def process(
        self, input_gen: Generator[Any, None, None]
    ) -> Generator[dict, None, None]:
        for obj in input_gen:
            if self.filter_func(obj):
                yield obj
