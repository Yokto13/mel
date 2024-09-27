from typing import Generator, List

from tokenization.pipeline.loggers.base import LoggerStep


def test_logger_logs_each_item():
    logged_items = []

    class TestLogger(LoggerStep):
        def logging_func(self, item: str) -> None:
            logged_items.append(item)

    logger = TestLogger()

    items = ["item1", "item2", "item3"]
    list(logger.process(iter(items)))

    assert logged_items == items


def test_logger_yields_each_item():
    class TestLogger(LoggerStep):
        def logging_func(self, item: str) -> None:
            pass

    logger = TestLogger()

    items = ["item1", "item2", "item3"]
    output_items = list(logger.process(iter(items)))

    assert output_items == items


def test_logger_calls_introduce_and_goodbye():
    class TestLogger(LoggerStep):
        def logging_func(self, item: str) -> None:
            pass

        def introduce(self) -> None:
            nonlocal introduce_called
            introduce_called = True

        def goodbye(self) -> None:
            nonlocal goodbye_called
            goodbye_called = True

    introduce_called = False
    goodbye_called = False

    logger = TestLogger()
    list(logger.process(iter(["item"])))

    assert introduce_called
    assert goodbye_called
