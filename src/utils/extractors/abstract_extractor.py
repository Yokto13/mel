from abc import ABC, abstractmethod

class AbstractExtractor(ABC):
    @abstractmethod
    def extract(self, dest):
        pass
