from abc import ABC, abstractmethod
from collections.abc import Generator


class PipelineStep(ABC):
    @abstractmethod
    def process(
        self, input_gen: Generator[str, None, None] = None
    ) -> Generator[str, None, None]:
        pass


class Pipeline(PipelineStep):
    def __init__(self):
        self.steps: list[PipelineStep] = []

    def add(self, step: PipelineStep) -> None:
        self.steps.append(step)

    def process(
        self, input_gen: Generator[str, None, None] = None
    ) -> Generator[str, None, None]:
        if input_gen is None:
            input_gen = self.steps[0].process()
        for step in self.steps[1:]:
            input_gen = step.process(input_gen)
        yield from input_gen

    def run(self) -> None:
        gen = self.process()
        for _ in gen:
            pass

    def __str__(self) -> str:
        return "\n".join(
            [
                "Tokenization Pipeline Steps:",
                *[
                    f"{i}. {step.__class__.__name__}"
                    for i, step in enumerate(self.steps, 1)
                ],
            ]
        )
