import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class Reranker:
    """Lightweight wrapper around Qwen reranker for mention-description scoring."""

    _DEFAULT_INSTRUCTION = (
        "Your task is to determine if the provided Wikipedia description correctly corresponds "
        "to the entity mention found in the query. The entity mention is marked by <M> and </M>. "
        "Check if the description matches the entity. Answer strictly with 'yes' or 'no'.\n"
        "Example:\n"
        "  Query: 'What is the capital of <M>France</M>?'\n"
        "  Description: 'Paris is the capital and largest city of France...'\n"
        "  Answer: no\n"
        "  Query: 'What is the <M>capital</M> of France?'\n"
        "  Description: 'Paris is the capital and largest city of France...'\n"
        "  Answer: yes"
    )
    _SYSTEM_PROMPT = (
        "<|im_start|>system\n"
        "Judge whether the Document meets the requirements based on the Query and the Instruct "
        'provided. Note that the answer can only be "yes" or "no".<|im_end|>\n'
        "<|im_start|>user\n"
    )
    _ASSISTANT_SUFFIX = "<|im_end|>\n" "<|im_start|>assistant\n" "<think>\n\n</think>\n\n"

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-Reranker-0.6B",
        max_length: int = 8192,
        instruction: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.max_length = max_length
        self.instruction = instruction or self._DEFAULT_INSTRUCTION

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
        self.model = AutoModelForCausalLM.from_pretrained(model_name).eval()

        self.token_false_id = self.tokenizer.convert_tokens_to_ids("no")
        self.token_true_id = self.tokenizer.convert_tokens_to_ids("yes")
        self.prefix_tokens = self.tokenizer.encode(self._SYSTEM_PROMPT, add_special_tokens=False)
        self.suffix_tokens = self.tokenizer.encode(self._ASSISTANT_SUFFIX, add_special_tokens=False)

    def score(self, mention: str, description: str, instruction: str | None = None) -> float:
        """Return probability that description matches mention."""
        formatted_query = self._format_query(mention)
        formatted_instruction = instruction or self.instruction
        prompt = self._format_instruction(formatted_instruction, formatted_query, description)
        inputs = self._process_inputs([prompt])
        probabilities = self._compute_probabilities(inputs)
        return probabilities[0]

    def _format_query(self, mention: str) -> str:
        has_markers = "<M>" in mention and "</M>" in mention
        wrapped = mention if has_markers else f"<M>{mention}</M>"
        return f"Identify the entity referenced by {wrapped}."

    def _format_instruction(self, instruction: str, query: str, document: str) -> str:
        return "<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}".format(
            instruction=instruction,
            query=query,
            doc=document,
        )

    def _process_inputs(self, prompts: list[str]):
        tokenized = self.tokenizer(
            prompts,
            padding=False,
            truncation="longest_first",
            return_attention_mask=False,
            max_length=self.max_length - len(self.prefix_tokens) - len(self.suffix_tokens),
        )

        for i, ids in enumerate(tokenized["input_ids"]):
            tokenized["input_ids"][i] = self.prefix_tokens + ids + self.suffix_tokens

        tokenized = self.tokenizer.pad(
            tokenized, padding=True, return_tensors="pt", max_length=self.max_length
        )

        for key in tokenized:
            tokenized[key] = tokenized[key].to(self.model.device)

        return tokenized

    @torch.no_grad()
    def _compute_probabilities(self, inputs):
        logits = self.model(**inputs).logits[:, -1, :]
        true_vector = logits[:, self.token_true_id]
        false_vector = logits[:, self.token_false_id]
        stacked = torch.stack([false_vector, true_vector], dim=1)
        log_probs = torch.nn.functional.log_softmax(stacked, dim=1)
        return log_probs[:, 1].exp().tolist()
