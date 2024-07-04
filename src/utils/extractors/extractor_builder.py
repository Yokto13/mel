from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable, Optional, Union

from utils.extractors.abstract_entry_processor import AbstractEntryProcessor
from utils.extractors.abstract_extractor import AbstractExtractor
from utils.extractors.damuel.damuel_iterator import DamuelExtractor
from utils.extractors.damuel.descriptions.descriptions_iterator import (
    DamuelDescriptionsIterator,
)
from utils.extractors.data_type import DataType


class ExtractorBuilder(ABC):
    def __init__(self) -> None:
        self._product: Optional[AbstractExtractor] = None
        self.reset()

    @abstractmethod
    def get_extractor(self) -> AbstractExtractor:
        pass

    @abstractmethod
    def reset(self):
        pass

    def set_source(self, source: Union[str, Path]):
        source = ExtractorBuilder._enusre_path_obj(source)
        self._product.source = source

    def set_entry_processor(self, processor):
        self._product = ExtractorWrapper(self._product, processor)

    @classmethod
    def _enusre_path_obj(cls, str_path):
        if isinstance(str_path, str):
            str_path = Path(str_path)
        return str_path


class ExtractorWrapper:
    # TODO: think of better way of doing this.
    def __init__(self, original, f):
        # This is needed to sidestep the overriden __setattr__ method
        super().__setattr__("_original", original)
        self.__extractor_wrapper_f = f

    def __iter__(self):
        return map(self.__extractor_wrapper_f, iter(self._original))

    def __getattr__(self, name):
        attr = getattr(self._original, name)
        if callable(attr):

            def method(*args, **kwargs):
                return attr(*args, **kwargs)

            return method
        return attr

    def __setattr__(self, name, value):
        if hasattr(self._original, name):
            setattr(self._original, name, value)
        else:
            super().__setattr__(name, value)

    def __repr__(self):
        return repr(self._original)


class DamuelExtractorBuilder(ExtractorBuilder):
    def set_file_acceptor(self, file_acceptor: Callable[[str], bool]):
        self._product.file_acceptor = file_acceptor

    def set_modulo_file_acceptor(self, modulo: int, remainder: int):
        def file_acceptor(file_name):
            return int(file_name.split("-")[-1]) % modulo == remainder

        self.set_file_acceptor(file_acceptor)

    def get_extractor(self) -> DamuelExtractor:
        pass


class DamuelDescriptionsExtractorBuilder(DamuelExtractorBuilder):
    def get_extractor(self) -> DamuelDescriptionsIterator:
        return self._product

    def reset(self):
        self._product = DamuelDescriptionsIterator()


class EntryProcessorBuilder(ABC):
    def __init__(self) -> None:
        self._product: Optional[AbstractExtractor] = None
        self.reset()

    @abstractmethod
    def get_processor(self) -> AbstractEntryProcessor:
        pass

    @abstractmethod
    def reset(self):
        pass

    def set_size(self, size: int):
        self._product.output_size = size
