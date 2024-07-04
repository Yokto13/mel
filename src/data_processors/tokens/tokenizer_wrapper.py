class TokenizerWrapper:
    def __init__(self, tokenizer, expected_size):
        self.tokenizer = tokenizer
        self.expected_size = expected_size

    def tokenize(self, text: str, max_length: int = None):
        tokens = self.tokenizer(
            text,
            return_tensors="np",
            padding="max_length",
            truncation=True,
            max_length=max_length or self.expected_size,
        )
        assert len(tokens["input_ids"]) <= self.expected_size
        return tokens["input_ids"][0]
