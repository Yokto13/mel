from tokenization.pipeline.loggers.basic import (
    _identity,
    CountingLogger,
    StatisticsLogger,
)


def test_counting_logger_counts_each_item(capsys):
    logger = CountingLogger()

    items = ["item1", "item2", "item3"]
    list(logger.process(iter(items)))

    assert logger.count == len(items)

    captured = capsys.readouterr()
    for i, item in enumerate(items, start=1):
        assert f"Processed item {i}: {item}" in captured.out


def test_counting_logger_yields_each_item():
    logger = CountingLogger()

    items = ["item1", "item2", "item3"]
    output_items = list(logger.process(iter(items)))

    assert output_items == items


def test_statistics_logger_logs_accessed_item(capsys):
    def item_accessor(item):
        return item["key"]

    logger = StatisticsLogger(item_accessor)

    items = [{"key": "item1"}, {"key": "item2"}, {"key": "item1"}]
    list(logger.process(iter(items)))

    logger.goodbye()

    captured = capsys.readouterr()
    assert "item1: 2" in captured.out
    assert "item2: 1" in captured.out


def test_statistics_logger_logs_item_by_index(capsys):
    def item_accessor(item):
        return item[1]

    logger = StatisticsLogger(item_accessor)

    items = [("a", "item1"), ("b", "item2"), ("c", "item1")]
    list(logger.process(iter(items)))

    logger.goodbye()

    captured = capsys.readouterr()
    assert "item1: 2" in captured.out
    assert "item2: 1" in captured.out


def test_statistics_logger_uses_identity_by_default(capsys):
    logger = StatisticsLogger()

    items = ["item1", "item2", "item1"]
    list(logger.process(iter(items)))

    logger.goodbye()

    captured = capsys.readouterr()
    assert "item1: 2" in captured.out
    assert "item2: 1" in captured.out


def test_statistics_logger_uses_provided_accessor(capsys):
    logger = StatisticsLogger(item_accessor=lambda x: x[0])

    items = [("item1", "a"), ("item2", "b"), ("item1", "c")]
    list(logger.process(iter(items)))

    logger.goodbye()

    captured = capsys.readouterr()
    assert "item1: 2" in captured.out
    assert "item2: 1" in captured.out
