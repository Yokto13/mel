import multiprocessing
from typing import List

from transformers import AutoTokenizer
import os
from tokenization.pipeline.pipelines import (
    DamuelDescriptionContextPipeline,
    DamuelLinkContextPipeline,
    MewsliMentionContextPipeline,
    Pipeline,
    DamuelDescriptionMentionPipeline,
    MewsliMentionPipeline,
)


def run_pipeline(pipeline: Pipeline) -> None:
    pipeline.run()


def run_pipelines(
    pipelines: List[Pipeline], num_processes: int = multiprocessing.cpu_count()
) -> None:
    with multiprocessing.Pool(processes=num_processes) as pool:
        pool.map(run_pipeline, pipelines)


def run_mewsli_mention() -> None:

    tokenizer = AutoTokenizer.from_pretrained(
        "/lnet/work/home-students-external/farhan/troja/outputs/models/LEALLA-base"
    )
    expected_size = 64

    output_base_dir = "/lnet/work/home-students-external/farhan/troja/outputs/pipelines/mewsli_mention"
    languages = ["ar", "de", "en", "es", "fa", "ja", "sr", "ta", "tr"]

    for lang in languages:
        os.makedirs(os.path.join(output_base_dir, lang), exist_ok=True)

    pipelines = [
        MewsliMentionPipeline(
            mewsli_tsv_path=f"/lnet/work/home-students-external/farhan/mewsli/mewsli-9/output/dataset/{lang}/mentions.tsv",
            tokenizer=tokenizer,
            expected_size=expected_size,
            output_filename=f"{output_base_dir}/{lang}/tokens_qids.npz",
            compress=True,
        )
        for lang in languages
    ]

    num_processes = 9
    run_pipelines(pipelines, num_processes)


def run_mewsli_context() -> None:

    tokenizer = AutoTokenizer.from_pretrained(
        "/lnet/work/home-students-external/farhan/troja/outputs/models/LEALLA-base"
    )
    expected_size = 64

    output_base_dir = "/lnet/work/home-students-external/farhan/troja/outputs/pipelines/mewsli_finetuning"
    languages = ["ar", "de", "en", "es", "fa", "ja", "sr", "ta", "tr"]

    for lang in languages:
        os.makedirs(os.path.join(output_base_dir, lang), exist_ok=True)

    pipelines = [
        MewsliMentionContextPipeline(
            mewsli_tsv_path=f"/lnet/work/home-students-external/farhan/mewsli/mewsli-9/output/dataset/{lang}/mentions.tsv",
            tokenizer=tokenizer,
            expected_size=expected_size,
            output_filename=f"{output_base_dir}/{lang}/tokens_qids.npz",
            compress=True,
        )
        for lang in languages
    ]

    num_processes = 9
    run_pipelines(pipelines, num_processes)


def run_damuel_description_mention() -> None:
    tokenizer = AutoTokenizer.from_pretrained(
        "/lnet/work/home-students-external/farhan/troja/outputs/models/LEALLA-base"
    )
    expected_size = 64

    output_base_dir = "/lnet/work/home-students-external/farhan/troja/outputs/pipelines/damuel_description_mention"
    languages = ["es"]

    for lang in languages:
        os.makedirs(os.path.join(output_base_dir, lang), exist_ok=True)

    pipelines = [
        DamuelDescriptionMentionPipeline(
            damuel_path=f"/lnet/work/home-students-external/farhan/damuel/1.0-xz/damuel_1.0_{lang}",
            tokenizer=tokenizer,
            expected_size=expected_size,
            output_filename=f"{output_base_dir}/{lang}/tokens_qids_{i}.npz",
            compress=True,
            remainder=i,
            mod=10,
        )
        for lang in languages
        for i in range(10)
    ]

    num_processes = 9
    run_pipelines(pipelines, num_processes)


def run_damuel_description_context() -> None:
    tokenizer = AutoTokenizer.from_pretrained(
        "/lnet/work/home-students-external/farhan/troja/outputs/models/LEALLA-base"
    )
    expected_size = 64

    output_base_dir = "/lnet/work/home-students-external/farhan/troja/outputs/pipelines/damuel_description_context"
    languages = ["es"]

    for lang in languages:
        os.makedirs(os.path.join(output_base_dir, lang), exist_ok=True)

    pipelines = [
        DamuelDescriptionContextPipeline(
            damuel_path=f"/lnet/work/home-students-external/farhan/damuel/1.0-xz/damuel_1.0_{lang}",
            tokenizer=tokenizer,
            expected_size=expected_size,
            output_filename=f"{output_base_dir}/{lang}/tokens_qids_{i}.npz",
            label_token="[M]",
            compress=True,
            remainder=i,
            mod=10,
        )
        for lang in languages
        for i in range(10)
    ]

    num_processes = 9
    run_pipelines(pipelines, num_processes)


def run_damuel_link_context() -> None:
    tokenizer = AutoTokenizer.from_pretrained(
        "/lnet/work/home-students-external/farhan/troja/outputs/models/LEALLA-base"
    )
    expected_size = 64

    output_base_dir = "/lnet/work/home-students-external/farhan/troja/outputs/pipelines/damuel_link_context"
    languages = ["es"]

    for lang in languages:
        os.makedirs(os.path.join(output_base_dir, lang), exist_ok=True)

    pipelines = [
        DamuelLinkContextPipeline(
            damuel_path=f"/lnet/work/home-students-external/farhan/damuel/1.0-xz/damuel_1.0_{lang}",
            tokenizer=tokenizer,
            expected_size=expected_size,
            output_filename=f"{output_base_dir}/{lang}/tokens_qids_{i}.npz",
            label_token="[M]",
            compress=True,
            remainder=i,
            mod=90,
        )
        for lang in languages
        for i in range(90)
    ]

    num_processes = 90
    run_pipelines(pipelines, num_processes)


if __name__ == "__main__":
    run_mewsli_mention()
