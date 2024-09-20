import multiprocessing
from typing import List

from tokenization.pipeline.pipeline import TokenizationPipeline


def run_pipeline(pipeline: TokenizationPipeline) -> None:
    pipeline.run()


def run_pipelines(
    pipelines: List[TokenizationPipeline], num_processes: int = None
) -> None:
    if num_processes is None:
        num_processes = multiprocessing.cpu_count()

    with multiprocessing.Pool(processes=num_processes) as pool:
        pool.map(run_pipeline, pipelines)


if __name__ == "__main__":
    # Example usage
    from transformers import AutoTokenizer
    from tokenization.pipeline.pipeline import MewsliMentionPipeline

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    expected_size = 64

    pipelines = [
        MewsliMentionPipeline(
            mewsli_tsv_path="path/to/mewsli1.tsv",
            tokenizer=tokenizer,
            expected_size=expected_size,
            output_filename="output1.npz",
            compress=True,
        ),
        MewsliMentionPipeline(
            mewsli_tsv_path="path/to/mewsli2.tsv",
            tokenizer=tokenizer,
            expected_size=expected_size,
            output_filename="output2.npz",
            compress=True,
        ),
        # Add more pipelines as needed
    ]

    num_processes = 4  # Specify the number of processes to use
    run_pipelines(pipelines, num_processes)
