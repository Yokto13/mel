from collections import defaultdict
from typing import Any

import numpy as np
from torch.utils.data import IterableDataset


class OnlyOnceDataset(IterableDataset):
    def __init__(self, iterable_dataset: IterableDataset):
        self._iterable_dataset = iterable_dataset
        self._db = _OnlyOnceTokens()

    def __iter__(self):
        for toks, qids in self._iterable_dataset:
            if (res := self._db(toks)) is not None:
                yield res, qids


class _OnlyOnceTokens:
    def __init__(self) -> None:
        self.memory = defaultdict(list)
        self._hasher = None

    def __call__(self, toks):
        if self._hasher is None:
            self._init_hasher(len(toks))

        h = self._hasher(len(toks))

        if h not in self.memory or not self._toks_in_memory(h, toks):
            self._add(h, toks)
            return toks
        return None

    def _init_hasher(self, cnt):
        self._hasher = _TokensHasher(cnt)

    def _add(self, h, toks):
        self.memory[h].append(toks)

    def _toks_in_memory(self, h, toks):
        return any(np.array_equal(toks, x) for x in self.memory[h])


class _TokensHasher:
    def __init__(self, sz, P=int(10**9 + 7), a=3) -> None:
        self.P = P
        self.a = a
        self.powers = None
        self._init_powers(sz)

    def __call__(self, toks):
        # won't overflow because power are uint64
        # could probably by done faster (the modulo part)
        res = self.powers * toks
        print(self.powers, toks, res)
        res %= self.P

        h = 0
        for x in res:
            print(h, x)
            if x == 0:
                return h
            h += x
            h %= self.P
        return h

    def _init_powers(self, cnt):
        self.powers = np.ones(cnt, dtype=np.int64)
        for i in range(1, cnt):
            self.powers[i] = (self.powers[i - 1] * self.a) % self.P
