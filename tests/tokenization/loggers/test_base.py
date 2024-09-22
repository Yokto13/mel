from typing import List

from tokenization.pipeline.loggers.base import LoggerStep


class TestLogger(LoggerStep):
    def __init__(self):
        super().__init__()
        self.logged_items: List[str] = []

    def logging_func(self, item: str) -> None:
        self.logged_items.append(item)


def test_logger_logs_each_item():
    logger = TestLogger()

    items = ["item1", "item2", "item3"]
    list(logger.process(iter(items)))

    assert logger.logged_items == items


def test_logger_yields_each_item():
    logger = TestLogger()

    items = ["item1", "item2", "item3"]
    output_items = list(logger.process(iter(items)))

    assert output_items == items
