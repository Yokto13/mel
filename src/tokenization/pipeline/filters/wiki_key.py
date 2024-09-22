from .base import Filter


class WikiKeyFilter(Filter):
    def __init__(self):
        super().__init__(self._filter_func)

    def _filter_func(self, entry: dict) -> bool:
        return "wiki" in entry
