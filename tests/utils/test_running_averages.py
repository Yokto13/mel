from collections import deque

import pytest

from utils.running_averages import RunningAverages


@pytest.fixture
def running_averages():
    return RunningAverages(running_average_small=3, running_average_big=5)


def test_initial_state(running_averages):
    assert running_averages.loss_running == deque(maxlen=3)
    assert running_averages.loss_running_big == deque(maxlen=5)
    assert running_averages.recall_running_1 == deque(maxlen=3)
    assert running_averages.recall_running_10 == deque(maxlen=3)
    assert running_averages.recall_running_1_big == deque(maxlen=5)
    assert running_averages.recall_running_10_big == deque(maxlen=5)


def test_update_loss(running_averages):
    running_averages.update_loss(0.5)
    assert list(running_averages.loss_running) == [0.5]
    assert list(running_averages.loss_running_big) == [0.5]
    assert running_averages.loss == 0.5
    assert running_averages.loss_big == 0.5


def test_update_recall(running_averages):
    running_averages.update_recall(0.1, 0.2)
    assert list(running_averages.recall_running_1) == [0.1]
    assert list(running_averages.recall_running_10) == [0.2]
    assert list(running_averages.recall_running_1_big) == [0.1]
    assert list(running_averages.recall_running_10_big) == [0.2]
    assert running_averages.recall_1 == 0.1
    assert running_averages.recall_10 == 0.2
    assert running_averages.recall_1_big == 0.1
    assert running_averages.recall_10_big == 0.2


def test_update_all(running_averages):
    running_averages.update_all(0.5, 0.1, 0.2)
    assert list(running_averages.loss_running) == [0.5]
    assert list(running_averages.loss_running_big) == [0.5]
    assert list(running_averages.recall_running_1) == [0.1]
    assert list(running_averages.recall_running_10) == [0.2]
    assert list(running_averages.recall_running_1_big) == [0.1]
    assert list(running_averages.recall_running_10_big) == [0.2]
    assert running_averages.loss == 0.5
    assert running_averages.loss_big == 0.5
    assert running_averages.recall_1 == 0.1
    assert running_averages.recall_10 == 0.2
    assert running_averages.recall_1_big == 0.1
    assert running_averages.recall_10_big == 0.2


def test_averages_with_multiple_updates(running_averages):
    running_averages.update_all(0.5, 0.1, 0.2)
    running_averages.update_all(0.7, 0.2, 0.3)
    running_averages.update_all(0.9, 0.3, 0.4)
    running_averages.update_all(
        1.1, 0.4, 0.5
    )  # This should push out the first entries in small deques

    assert running_averages.loss == (0.7 + 0.9 + 1.1) / 3
    assert running_averages.recall_1 == (0.2 + 0.3 + 0.4) / 3
    assert running_averages.recall_10 == (0.3 + 0.4 + 0.5) / 3

    assert running_averages.loss_big == (0.5 + 0.7 + 0.9 + 1.1) / 4
    assert running_averages.recall_1_big == (0.1 + 0.2 + 0.3 + 0.4) / 4
    assert running_averages.recall_10_big == (0.2 + 0.3 + 0.4 + 0.5) / 4


def test_empty_deques(running_averages):
    assert running_averages.loss == 0.0
    assert running_averages.recall_1 == 0.0
    assert running_averages.recall_10 == 0.0
    assert running_averages.loss_big == 0.0
    assert running_averages.recall_1_big == 0.0
    assert running_averages.recall_10_big == 0.0


if __name__ == "__main__":
    pytest.main()
