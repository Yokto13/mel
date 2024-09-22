from typing import List

from tokenization.pipeline.loggers.base import Logger


def test_logger_logs_each_item():
    logged_items: List[str] = []

    def logging_func(item: str) -> None:
        logged_items.append(item)

    logger = Logger(logging_func)

    items = ["item1", "item2", "item3"]
    list(logger.process(iter(items)))

    assert logged_items == items


def test_logger_yields_each_item():
    def logging_func(item: str) -> None:
        pass

    logger = Logger(logging_func)

    items = ["item1", "item2", "item3"]
    output_items = list(logger.process(iter(items)))

    assert output_items == items
