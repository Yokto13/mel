from collections import deque


class RunningAverages:
    def __init__(self, running_average_small, running_average_big):
        self.loss_running = self._create_deque(running_average_small)
        self.loss_running_big = self._create_deque(running_average_big)
        self.recall_running_1 = self._create_deque(running_average_small)
        self.recall_running_10 = self._create_deque(running_average_small)
        self.recall_running_1_big = self._create_deque(running_average_big)
        self.recall_running_10_big = self._create_deque(running_average_big)

    def _create_deque(self, maxlen):
        return deque(maxlen=maxlen)

    def update_loss(self, loss):
        self._update_deques(loss, self.loss_running, self.loss_running_big)

    def update_recall(self, recall_1, recall_10):
        self._update_deques(recall_1, self.recall_running_1, self.recall_running_1_big)
        self._update_deques(
            recall_10, self.recall_running_10, self.recall_running_10_big
        )

    def update_all(self, loss, recall_1, recall_10):
        self.update_loss(loss)
        self.update_recall(recall_1, recall_10)

    def _update_deques(self, value, small_deque, big_deque):
        small_deque.append(value)
        big_deque.append(value)

    def _calculate_average(self, deque):
        return sum(deque) / len(deque) if deque else 0.0

    @property
    def loss(self):
        return self._calculate_average(self.loss_running)

    @property
    def recall_1(self):
        return self._calculate_average(self.recall_running_1)

    @property
    def recall_10(self):
        return self._calculate_average(self.recall_running_10)

    @property
    def loss_big(self):
        return self._calculate_average(self.loss_running_big)

    @property
    def recall_1_big(self):
        return self._calculate_average(self.recall_running_1_big)

    @property
    def recall_10_big(self):
        return self._calculate_average(self.recall_running_10_big)
