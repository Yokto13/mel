from collections.abc import Iterable
from pathlib import Path


class AbstractExtractor(Iterable):
    def __init__(self) -> None:
        self.source: Path = None
