class PreprocessingDict:
    def __init__(self, *args, **kwargs):
        self._dict = dict(*args, **kwargs)

    def preprocess_key(self, key):
        # This method should be overridden in a subclass to implement
        # custom key preprocessing logic
        return key

    def __getitem__(self, key):
        preprocessed_key = self.preprocess_key(key)
        return self._dict[preprocessed_key]

    def __setitem__(self, key, value):
        preprocessed_key = self.preprocess_key(key)
        self._dict[preprocessed_key] = value

    def __delitem__(self, key):
        preprocessed_key = self.preprocess_key(key)
        del self._dict[preprocessed_key]

    def __contains__(self, key):
        preprocessed_key = self.preprocess_key(key)
        return preprocessed_key in self._dict

    def __len__(self):
        return len(self._dict)

    def __iter__(self):
        return iter(self._dict)

    def __repr__(self):
        return f"{self.__class__.__name__}({self._dict})"

    def get(self, key, default=None):
        preprocessed_key = self.preprocess_key(key)
        return self._dict.get(preprocessed_key, default)

    def pop(self, key, default=None):
        preprocessed_key = self.preprocess_key(key)
        return self._dict.pop(preprocessed_key, default)

    def update(self, *args, **kwargs):
        if args:
            if len(args) > 1:
                raise TypeError(
                    "update expected at most 1 argument, got {}".format(len(args))
                )
            other = dict(args[0])
            for key, value in other.items():
                self[key] = value
        for key, value in kwargs.items():
            self[key] = value

    def keys(self):
        return self._dict.keys()

    def values(self):
        return self._dict.values()

    def items(self):
        return self._dict.items()
