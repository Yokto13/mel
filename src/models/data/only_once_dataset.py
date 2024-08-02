from collections import defaultdict
from typing import Any

import numpy as np
from torch.utils.data import IterableDataset


class OnlyOnceDataset(IterableDataset):
    def __init__(self, iterable_dataset: IterableDataset):
        self._iterable_dataset = iterable_dataset
        # self._db = OnlyOnceTokens()
        self._db = set()

    def __iter__(self):
        for toks, qids in self._iterable_dataset:
            # tolist first provides 2-3x times speed up when converting to tuple
            tup_toks = tuple(toks.tolist())
            if tup_toks not in self._db:
                self._db.add(tup_toks)
                yield toks, qids
            # if (res := self._db(toks)) is not None:
            # yield res, qids


class OnlyOnceTokens:
    def __init__(self) -> None:
        self.memory = self.init_memory()
        self.hasher = None

    def process_hash(self, h, toks):
        if self.is_missing(h, toks):
            self.add(h, toks)
            return toks
        return None

    def init_memory(self):
        return defaultdict(list)

    def add(self, h, toks):
        self.memory[h].append(toks)

    def is_missing(self, h, toks):
        return h not in self.memory or not self.toks_in_memory(h, toks)

    def toks_in_memory(self, h, toks):
        return any(np.array_equal(toks, x) for x in self.memory[h])

    def __call__(self, toks):
        if self.hasher is None:
            self._init_hasher(len(toks))

        h = self.hasher(toks)

        return self.process_hash(h, toks)

    def _init_hasher(self, cnt):
        self.hasher = _TokensHasher(cnt)


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
        res %= self.P

        h = 0
        for x in res:
            if x == 0:
                return h
            h += x
            h %= self.P
        return h

    def _init_powers(self, cnt):
        self.powers = np.ones(cnt, dtype=np.int64)
        for i in range(1, cnt):
            self.powers[i] = (self.powers[i - 1] * self.a) % self.P
