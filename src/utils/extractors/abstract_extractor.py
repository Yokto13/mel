from abc import ABC
from pathlib import Path


class AbstractExtractor(ABC.Iterator):
    def __init__(self) -> None:
        self.source: Path = None
