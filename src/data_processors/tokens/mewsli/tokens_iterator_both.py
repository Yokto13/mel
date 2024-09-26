from collections import defaultdict

from data_processors.tokens.mewsli.tokens_iterator import MewsliTokensIterator


class MewsliTokensIteratorBoth:
    def __init__(self, *args, **kwargs):
        self.mention_iterator = MewsliTokensIterator(*args, use_context=False, **kwargs)
        self.context_iterator = MewsliTokensIterator(*args, use_context=True, **kwargs)
        self.qid_occurrences = defaultdict(int)

    def __iter__(self):
        for mention, context in zip(self.mention_iterator, self.context_iterator):
            yield mention, context
