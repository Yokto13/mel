from collections.abc import Generator
from ..base import PipelineStep


class ChainStep(PipelineStep):
    def __init__(self, steps):
        super().__init__()
        self.steps = steps

    def process(
        self, input_gen: Generator[str, None, None] = None
    ) -> Generator[str, None, None]:
        if input_gen is not None:
            raise ValueError("ChainStep does not support input generator")
        for step in self.steps:
            gen = step.process(None)
            yield from gen
