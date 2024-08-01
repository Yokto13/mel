from collections import defaultdict

import numpy as np
from models.data.tokens_multiplicity_counter import TokensMultiplicityCounter


class TokensQidsMultiplicityCounter(TokensMultiplicityCounter):
    """Counter for tokens.

    TODO: unintuitive to use. Consider refactoring it (especially interface).
    """

    def __init__(self) -> None:
        super().__init__()

    def process_hash(self, h, dato):
        if self.is_missing(h, dato):
            self.add(h, dato)
        self.increase(h, dato)
        return self._count(h, dato)

    def add(self, h, dato):
        self.memory[h].append([dato, 0])

    def toks_in_memory(self, h, dato):
        return any(
            (dato[1] == x[0][1] and np.array_equal(dato[0], x[0][0]))
            for x in self.memory[h]
        )

    def increase(self, h, dato):
        self.memory[h][self._get_index_in_cell_h(h, dato)][1] += 1

    def _count(self, h, dato):
        return self.memory[h][self._get_index_in_cell_h(h, dato)][1]

    def _get_index_in_cell_h(self, h, dato):
        toks = dato[0]
        qid = dato[1]
        idx_after_h = 0
        if len(self.memory[h]) > 1:
            # we need to find toks
            idx_after_h = None
            for i in range(len(self.memory[h])):
                if qid == self.memory[h][i][0][1] and np.array_equal(
                    self.memory[h][i][0][0], toks
                ):
                    idx_after_h = i
                    break
        return idx_after_h
