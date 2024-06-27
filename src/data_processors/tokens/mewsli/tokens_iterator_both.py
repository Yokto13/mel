from collections import defaultdict

from data_processors.tokens.mewsli.tokens_iterator import MewsliTokensIterator


class MewsliTokensIteratorBoth:
    def __init__(self, *args, max_per_qid, **kwargs):
        self.mention_iterator = MewsliTokensIterator(*args, use_context=False, **kwargs)
        self.context_iterator = MewsliTokensIterator(*args, use_context=True, **kwargs)
        self.max_per_qid = max_per_qid
        self.qid_occurrences = defaultdict(int)

    def __iter__(self):
        for mention, context in zip(self.mention_iterator, self.context_iterator):
            if self.qid_occurrences[mention.qid] < self.max_per_qid:
                self.qid_occurrences[mention.qid] += 1
                yield mention, context
