from abc import ABC, abstractmethod
from collections.abc import Generator

from ..base import PipelineStep


class LoaderStep(PipelineStep, ABC):
    def __init__(self, path: str):
        self.path = path
