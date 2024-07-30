from collections import defaultdict

import numpy as np
from models.data.only_once_dataset import OnlyOnceTokens


class TokensMultiplicityCounter(OnlyOnceTokens):
    """Counter for tokens.

    TODO: unintuitive to use. Consider refactoring it (especially interface).
    """

    def __init__(self) -> None:
        super().__init__()

    def process_hash(self, h, toks):
        if self.is_missing(h, toks):
            self.add(h, toks)
        self.increase(h, toks)
        return self._count(h, toks)

    def add(self, h, toks):
        self.memory[h].append([toks, 0])

    def toks_in_memory(self, h, toks):
        return any(np.array_equal(toks, x[0]) for x in self.memory[h])

    def increase(self, h, toks):
        self.memory[h][self._get_index_in_cell_h(h, toks)][1] += 1

    def _count(self, h, toks):
        return self.memory[h][self._get_index_in_cell_h(h, toks)][1]

    def count(self, toks):
        h = self.hasher(toks)

    def _get_index_in_cell_h(self, h, toks):
        idx_after_h = 0
        if len(self.memory[h]) > 1:
            # we need to find toks
            idx_after_h = None
            for i in range(len(self.memory[h])):
                if np.array_equal(self.memory[h][i][0], toks):
                    idx_after_h = i
                    break
        return idx_after_h
