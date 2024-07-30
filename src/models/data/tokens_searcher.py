from collections import defaultdict
import numpy as np

from bisect import bisect_left, bisect_right


class TokensSearcher:
    def __init__(self, tokens: np.ndarray, metadata: np.ndarray):
        self.tokens = tokens

        # np.lexsort gives indices of lexicographical order but in which it consumes tokens is not obvious
        self.indices = np.lexsort(tokens.T[::-1])

        self.tokens = self.tokens[self.indices]

        self.metadata = metadata[self.indices]

    def find(self, query, save=False):
        lo, hi = 0, len(self.tokens)
        print("queyr, tokens", len(query), print(self.tokens))
        for level in range(query.shape[0]):
            if hi - lo == 1:
                if save:
                    if not np.array_equal(self.tokens[lo], query):
                        raise KeyError(f"{query} not present in the TokensSearcher.")
                return self.metadata[lo]
            print(f"looking for {query[level]}")
            lo = bisect_left(self.tokens, query[level], lo, hi, key=lambda x: x[level])
            hi = bisect_right(self.tokens, query[level], lo, hi, key=lambda x: x[level])
            print("lo hi", lo, hi)
        if hi - lo == 1:
            if save:
                if not np.array_equal(self.tokens[lo], query):
                    raise KeyError(f"{query} not present in the TokensSearcher.")
            return self.metadata[lo]
        raise KeyError(f"{query} not present in the TokensSearcher.")
