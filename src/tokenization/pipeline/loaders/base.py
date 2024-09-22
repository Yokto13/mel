from abc import ABC, abstractmethod
from collections.abc import Generator

from ..pipelines import PipelineStep


class LoaderStep(PipelineStep, ABC):
    def __init__(self, path: str):
        self.path = path

    @abstractmethod
    def run(self) -> Generator[str, None, None]:
        pass
