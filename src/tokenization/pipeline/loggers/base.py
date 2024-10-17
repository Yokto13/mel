from abc import ABC, abstractmethod
from collections.abc import Generator

from ..base import PipelineStep


class LoggerStep(PipelineStep, ABC):
    def __init__(self):
        super().__init__()

    def process(
        self, input_gen: Generator[str, None, None] = None
    ) -> Generator[str, None, None]:
        self.introduce()
        for item in input_gen:
            self.logging_func(item)
            yield item
        self.goodbye()

    @abstractmethod
    def logging_func(self, item: str) -> None:
        pass

    def introduce(self) -> None:
        pass

    def goodbye(self) -> None:
        pass
