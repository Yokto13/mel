from collections.abc import Callable
from pathlib import Path

from utils.extractors.output_type import OutputType


class AbstractEntryProcessor(Callable):
    def __init__(self) -> None:
        self.source: Path = None
        self.output_type: OutputType = None
        self.size: int = None
