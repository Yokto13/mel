from collections import Counter
from collections.abc import Callable
from typing import Any

from .base import LoggerStep


class CountingLogger(LoggerStep):
    def __init__(self):
        super().__init__()
        self.count = 0

    def logging_func(self, item: str) -> None:
        self.count += 1
        print(f"Processed item {self.count}: {item}")


def _identity(x: Any) -> Any:
    return x


class StatisticsLogger(LoggerStep):
    """
    A logger that collects statistics about the frequency of items.

    This logger uses a user-provided `item_accessor` function to extract a value
    from each item. It then counts the frequency of each unique accessed value.

    At the end of processing, the `goodbye` method prints the `most_common`
    accessed values and their frequencies.

    Args:
        item_accessor (Callable[[Any], Any], optional): A function that takes an item and returns a value to be counted.
            Defaults to `_identity`, which returns the item unchanged.
        most_common (int, optional): The number of most common items to print in the `goodbye` method.
            Defaults to 10.

    Attributes:
        item_accessor (Callable[[Any], Any]): The function used to extract a value from each item.
        statistics (Counter): A Counter object that keeps track of the frequency of each accessed value.
        most_common (int): The number of most common items to print in the `goodbye` method.
    """

    def __init__(
        self, item_accessor: Callable[[Any], Any] = _identity, most_common: int = 3
    ) -> None:
        super().__init__()
        self.item_accessor = item_accessor
        self.statistics = Counter()
        self.most_common = most_common

    def logging_func(self, item: Any) -> None:
        accessed_item = self.item_accessor(item)
        self.statistics[accessed_item] += 1

    def goodbye(self) -> None:
        print(f"Top {self.most_common} most occurring items:")
        for item, count in self.statistics.most_common(self.most_common):
            print(f"{item}: {count}")
