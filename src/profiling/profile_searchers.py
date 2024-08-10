import sys
from timeit import timeit
from typing import Callable

sys.path.append("../")

import numpy as np

from models.searcher import ScaNNSearcher, Searcher


class Profiler:
    def __init__(self, searcher_constructor: type[Searcher], cnt, dim) -> None:
        self.embs = np.random.random((cnt, dim))
        self.results = np.arange(cnt)
        self.searcher = searcher_constructor(self.embs, self.results)
        self.dim = dim

    def profile_known(self, queries_cnt, batch_size, num_neighbors, repeats=10) -> None:
        inds = np.random.permutation(len(self.embs))[: queries_cnt * batch_size]
        queries = self.embs[inds].reshape((queries_cnt, batch_size, self.dim))

        def test():
            for batch in queries:
                self.searcher.find(batch, num_neighbors)

        self._profile(test, repeats)

    def _profile(self, f: Callable, repeats: int) -> None:
        print(timeit(f, number=repeats))


if __name__ == "__main__":
    profiler = Profiler(ScaNNSearcher, int(10**7), 128)
    profiler.profile_known(100, 768, 801)
