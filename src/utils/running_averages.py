from collections import deque


class RunningAverages:
    def __init__(self, running_average_small, running_average_big):
        self.loss_running = deque(maxlen=running_average_small)
        self.loss_running_big = deque(maxlen=running_average_big)
        self.recall_running_1 = deque(maxlen=running_average_small)
        self.recall_running_10 = deque(maxlen=running_average_small)
        self.recall_running_1_big = deque(maxlen=running_average_big)
        self.recall_running_10_big = deque(maxlen=running_average_big)

    def update_loss(self, loss):
        self.loss_running.append(loss)
        self.loss_running_big.append(loss)

    def update_recall(self, recall_1, recall_10):
        self.recall_running_1.append(recall_1)
        self.recall_running_10.append(recall_10)
        self.recall_running_1_big.append(recall_1)
        self.recall_running_10_big.append(recall_10)

    def update_all(self, loss, recall_1, recall_10):
        self.update_loss(loss)
        self.update_recall(recall_1, recall_10)

    @property
    def loss(self):
        return sum(self.loss_running) / len(self.loss_running)

    @property
    def recall_1(self):
        return sum(self.recall_running_1) / len(self.recall_running_1)

    @property
    def recall_10(self):
        return sum(self.recall_running_10) / len(self.recall_running_10)

    @property
    def loss_big(self):
        return sum(self.loss_running_big) / len(self.loss_running_big)

    @property
    def recall_1_big(self):
        return sum(self.recall_running_1_big) / len(self.recall_running_1_big)

    @property
    def recall_10_big(self):
        return sum(self.recall_running_10_big) / len(self.recall_running_10_big)
