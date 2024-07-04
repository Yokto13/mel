from abc import ABC
from pathlib import Path

from utils.extractors.output_type import OutputType


class AbstractEntryProcessor(ABC.Callable):
    def __init__(self) -> None:
        self.source: Path = None
        self.output_type: OutputType = None
        self.size:int = None
