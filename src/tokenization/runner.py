import multiprocessing
from typing import List
import logging

import gin

from transformers import AutoTokenizer
import os
from tokenization.pipeline.pipelines import (
    DamuelDescriptionContextPipeline,
    DamuelLinkContextPipeline,
    MewsliContextPipeline,
    Pipeline,
    DamuelDescriptionMentionPipeline,
    MewsliMentionPipeline,
)

logging.basicConfig(level=logging.INFO)


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

    pipelines_dict = {
        lang: [
            MewsliMentionPipeline(
                mewsli_tsv_path=f"/lnet/work/home-students-external/farhan/mewsli/mewsli-9/output/dataset/{lang}/mentions.tsv",
                tokenizer=tokenizer,
                expected_size=expected_size,
                output_filename=f"{output_base_dir}/{lang}/tokens_qids.npz",
                compress=True,
            )
        ]
        for lang in languages
    }

    num_processes = 9
    for lang, pipelines in pipelines_dict.items():
        run_pipelines(pipelines, num_processes)
        logging.info(f"Finished processing language: {lang}")


def run_mewsli_context() -> None:
    tokenizer = AutoTokenizer.from_pretrained(
        "/lnet/work/home-students-external/farhan/troja/outputs/models/LEALLA-base"
    )
    expected_size = 64

    output_base_dir = "/lnet/work/home-students-external/farhan/troja/outputs/pipelines/mewsli_finetuning"
    languages = ["ar", "de", "en", "es", "fa", "ja", "sr", "ta", "tr"]

    for lang in languages:
        os.makedirs(os.path.join(output_base_dir, lang), exist_ok=True)

    pipelines_dict = {
        lang: [
            MewsliContextPipeline(
                mewsli_tsv_path=f"/lnet/work/home-students-external/farhan/mewsli/mewsli-9/output/dataset/{lang}/mentions.tsv",
                tokenizer=tokenizer,
                label_token="[M]",
                expected_size=expected_size,
                output_filename=f"{output_base_dir}/{lang}/tokens_qids.npz",
                compress=True,
            )
        ]
        for lang in languages
    }

    num_processes = 9
    for lang, pipelines in pipelines_dict.items():
        run_pipelines(pipelines, num_processes)
        logging.info(f"Finished processing language: {lang}")


def run_damuel_description_mention() -> None:
    tokenizer = AutoTokenizer.from_pretrained(
        "/lnet/work/home-students-external/farhan/troja/outputs/models/LEALLA-base"
    )
    expected_size = 64

    output_base_dir = "/lnet/work/home-students-external/farhan/troja/outputs/pipelines/damuel_description_mention"
    languages = ["es"]

    for lang in languages:
        os.makedirs(os.path.join(output_base_dir, lang), exist_ok=True)

    pipelines_dict = {
        lang: [
            DamuelDescriptionMentionPipeline(
                damuel_path=f"/lnet/work/home-students-external/farhan/damuel/1.0-xz/damuel_1.0_{lang}",
                tokenizer=tokenizer,
                expected_size=expected_size,
                output_filename=f"{output_base_dir}/{lang}/tokens_qids_{i}.npz",
                compress=True,
                remainder=i,
                mod=10,
            )
            for i in range(10)
        ]
        for lang in languages
    }

    num_processes = 9
    for lang, pipelines in pipelines_dict.items():
        run_pipelines(pipelines, num_processes)
        logging.info(f"Finished processing language: {lang}")


@gin.configurable
def run_damuel_description_context(
    model_path: str,
    expected_size: int,
    output_base_dir: str,
    languages: list[str],
    damuel_base_path: str,
    label_token: str,
    compress: bool,
    remainder_mod: int,
    num_processes: int,
) -> None:
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    for lang in languages:
        os.makedirs(os.path.join(output_base_dir, lang, "descs_pages"), exist_ok=True)

    pipelines_dict = {
        lang: [
            DamuelDescriptionContextPipeline(
                damuel_path=f"{damuel_base_path}/damuel_1.0_{lang}",
                tokenizer=tokenizer,
                expected_size=expected_size,
                output_filename=f"{output_base_dir}/{lang}/tokens_qids_{i}.npz",
                label_token=label_token,
                compress=compress,
                remainder=i,
                mod=remainder_mod,
            )
            for i in range(remainder_mod)
        ]
        for lang in languages
    }

    for lang, pipelines in pipelines_dict.items():
        run_pipelines(pipelines, num_processes)
        logging.info(f"Finished processing language: {lang}")


def run_damuel_link_context() -> None:
    tokenizer = AutoTokenizer.from_pretrained(
        "/lnet/work/home-students-external/farhan/troja/outputs/models/LEALLA-base"
    )
    expected_size = 64

    output_base_dir = "/lnet/work/home-students-external/farhan/troja/outputs/pipelines/damuel_link_context"
    languages = ["es"]

    for lang in languages:
        os.makedirs(os.path.join(output_base_dir, lang), exist_ok=True)

    pipelines_dict = {
        lang: [
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
            for i in range(90)
        ]
        for lang in languages
    }

    num_processes = 90
    for lang, pipelines in pipelines_dict.items():
        run_pipelines(pipelines, num_processes)
        logging.info(f"Finished processing language: {lang}")


if __name__ == "__main__":
    run_mewsli_mention()
