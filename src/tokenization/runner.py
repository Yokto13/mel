import glob
import logging
import multiprocessing
import os
from typing import Any, List

import gin
from tokenization.pipeline.pipelines import (
    DamuelDescriptionContextPipeline,
    DamuelDescriptionMentionPipeline,
    DamuelLinkContextPipeline,
    DamuelLinkMentionPipeline,
    MewsliContextPipeline,
    MewsliMentionPipeline,
    Pipeline,
    DamuelAliasTablePipeline,
)

from transformers import AutoTokenizer

logging.basicConfig(level=logging.INFO)


def run_pipeline(pipeline: Pipeline) -> None:
    pipeline.run()


def run_pipelines(
    pipelines: List[Pipeline], num_processes: int = multiprocessing.cpu_count()
) -> None:
    with multiprocessing.Pool(processes=num_processes) as pool:
        pool.map(run_pipeline, pipelines)


def process_pipeline(pipe: Pipeline) -> list[Any]:
    return list(pipe.process())


def process_pipelines(
    pipelines: List[Pipeline], num_processes: int = multiprocessing.cpu_count()
) -> None:
    with multiprocessing.Pool(processes=num_processes) as pool:
        result = pool.map(process_pipeline, pipelines)
    # print(result)
    return result


@gin.configurable
def run_alias_table_damuel(
    path: str,
    num_processes: int,
    remainder_mod: int,
) -> list[Any]:
    pipelines = [
        DamuelAliasTablePipeline(
            damuel_path=path, remainder=i, mod=remainder_mod, logger=None
        )
        for i in range(remainder_mod)
    ]
    results = process_pipelines(pipelines, num_processes)
    logging.info("Finished processing one language for DaMuEL Alias Table")
    return results


@gin.configurable
def run_mewsli_mention(
    model_path: str,
    expected_size: int,
    output_base_dir: str,
    languages: List[str],
    mewsli_dataset_path: str,
    compress: bool,
    num_processes: int,
) -> None:
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    for lang in languages:
        os.makedirs(os.path.join(output_base_dir, lang), exist_ok=True)

    pipelines = [
        MewsliMentionPipeline(
            mewsli_tsv_path=os.path.join(mewsli_dataset_path, lang, "mentions.tsv"),
            tokenizer=tokenizer,
            expected_size=expected_size,
            output_filename=f"{output_base_dir}/{lang}/tokens_qids.npz",
            compress=compress,
        )
        for lang in languages
    ]

    run_pipelines(pipelines, num_processes)
    logging.info("Finished processing all languages for Mewsli Mention")


@gin.configurable
def run_mewsli_context(
    model_path: str,
    expected_size: int,
    output_base_dir: str,
    languages: List[str],
    mewsli_dataset_path: str,
    label_token: str,
    compress: bool,
    num_processes: int,
) -> None:
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    for lang in languages:
        os.makedirs(os.path.join(output_base_dir, lang), exist_ok=True)

    pipelines = [
        MewsliContextPipeline(
            mewsli_tsv_path=os.path.join(mewsli_dataset_path, lang, "mentions.tsv"),
            tokenizer=tokenizer,
            label_token=label_token,
            expected_size=expected_size,
            output_filename=f"{output_base_dir}/{lang}/tokens_qids.npz",
            compress=compress,
        )
        for lang in languages
    ]

    run_pipelines(pipelines, num_processes)
    logging.info("Finished processing all languages for Mewsli Context")


@gin.configurable
def run_damuel_description_mention(
    model_path: str,
    expected_size: int,
    output_base_dir: str,
    languages: List[str],
    damuel_base_path: str,
    compress: bool,
    num_processes: int,
    remainder_mod: int,
    require_wiki_page: bool = True,
) -> None:
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    for lang in languages:
        os.makedirs(os.path.join(output_base_dir, lang), exist_ok=True)

    pipelines_dict = {
        lang: [
            DamuelDescriptionMentionPipeline(
                damuel_path=f"{damuel_base_path}/damuel_2.0-dev_{lang}",
                tokenizer=tokenizer,
                expected_size=expected_size,
                output_filename=f"{output_base_dir}/{lang}/tokens_qids_{i}.npz",
                compress=compress,
                remainder=i,
                mod=remainder_mod,
                require_wiki_page=require_wiki_page,
            )
            for i in range(remainder_mod)
        ]
        for lang in languages
    }

    for lang, pipelines in pipelines_dict.items():
        run_pipelines(pipelines, num_processes)
        logging.info(f"Finished processing language: {lang}")


@gin.configurable
def run_damuel_description_context(
    model_path: str,
    expected_size: int,
    output_base_dir: str,
    languages: List[str],
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
                damuel_path=f"{damuel_base_path}/damuel_2.0-dev_{lang}",
                tokenizer=tokenizer,
                expected_size=expected_size,
                output_filename=f"{output_base_dir}/{lang}/descs_pages/tokens_qids_{i}.npz",
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


@gin.configurable
def run_damuel_link_context(
    model_path: str,
    expected_size: int,
    output_base_dir: str,
    languages: List[str],
    damuel_base_path: str,
    label_token: str,
    compress: bool,
    remainder_mod: int,
    num_processes: int,
) -> None:
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    for lang in languages:
        os.makedirs(os.path.join(output_base_dir, lang, "links"), exist_ok=True)

    pipelines_dict = {
        lang: [
            DamuelLinkContextPipeline(
                damuel_path=f"{damuel_base_path}/damuel_2.0-dev_{lang}",
                tokenizer=tokenizer,
                expected_size=expected_size,
                output_filename=f"{output_base_dir}/{lang}/links/tokens_qids_{i}.npz",
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


@gin.configurable
def run_damuel_link_mention(
    model_path: str,
    expected_size: int,
    output_base_dir: str,
    languages: List[str],
    damuel_base_path: str,
    compress: bool,
    num_processes: int,
    require_link_wiki_origin: bool,
    remainder_mod: int,
) -> None:
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    for lang in languages:
        os.makedirs(os.path.join(output_base_dir, lang), exist_ok=True)

    pipelines_dict = {
        lang: [
            DamuelLinkMentionPipeline(
                damuel_path=f"{damuel_base_path}/damuel_2.0-dev_{lang}",
                tokenizer=tokenizer,
                expected_size=expected_size,
                output_filename=f"{output_base_dir}/{lang}/tokens_qids_{i}.npz",
                compress=compress,
                remainder=i,
                mod=remainder_mod,
                require_link_wiki_origin=require_link_wiki_origin,
            )
            for i in range(remainder_mod)
        ]
        for lang in languages
    }

    for lang, pipelines in pipelines_dict.items():
        run_pipelines(pipelines, num_processes)
        logging.info(f"Finished processing language: {lang}")


@gin.configurable
def run_damuel_mention(
    model_path: str,
    expected_size: int,
    output_base_dir: str,
    languages: List[str],
    damuel_base_path: str,
    compress: bool,
    num_processes: int,
    remainder_mod: int,
    require_link_wiki_origin: bool,
    require_wiki_page: bool,
) -> None:
    """
    Run the DaMuEL mention tokenization process.
    Tokenizes only mentions (no context).
    Usefull for baselines like OLPEAT.

    We have two sources of mentions:
    - titles and labels of wikipedia entities
    - text corresponding to a link from one page to another

    Args:
        model_path (str): Path to the pre-trained model for tokenization.
        expected_size (int): Expected size of the tokenized output.
        output_base_dir (str): Base directory for output files.
        languages (List[str]): List of languages to process.
        damuel_base_path (str): Base path for DaMuEL dataset.
        compress (bool): Whether to compress the output files.
        num_processes (int): Number of processes to use for parallel processing.
        remainder_mod (int): Modulo value for remainder-based processing.
        require_link_wiki_origin (bool): Whether to require links to originate from Wikipedia.
            DaMuEL adds some links that are not originally present in Wikipedia.
            If True, the links should correspond to the original Wikipedia dump, if False
            the result contains also added links. On one hand, the former gives us less mentions,
            on the other hand, we avoid including links of lower quality that might be created by the addition.
        require_wiki_page (bool): If True, only entities from language L that have a wikipedia page in L are considered.
            if False, all entites of language L are considered.
    """

    # Run description mention tokenization
    run_damuel_description_mention(
        model_path=model_path,
        expected_size=expected_size,
        output_base_dir=output_base_dir,
        languages=languages,
        damuel_base_path=damuel_base_path,
        compress=compress,
        num_processes=num_processes,
        remainder_mod=remainder_mod,
        require_wiki_page=require_wiki_page,
    )

    # Rename description mention files
    for lang in languages:
        lang_dir = os.path.join(output_base_dir, lang)
        for file in glob.glob(os.path.join(lang_dir, "tokens_qids_*.npz")):
            new_name = file.replace("tokens_qids_", "tokens_qids_descs_")
            os.rename(file, new_name)

    # Run link mention tokenization
    run_damuel_link_mention(
        model_path=model_path,
        expected_size=expected_size,
        output_base_dir=output_base_dir,
        languages=languages,
        damuel_base_path=damuel_base_path,
        compress=compress,
        num_processes=num_processes,
        require_link_wiki_origin=require_link_wiki_origin,
        remainder_mod=remainder_mod,
    )

    # Rename link mention files
    for lang in languages:
        lang_dir = os.path.join(output_base_dir, lang)
        for file in glob.glob(os.path.join(lang_dir, "tokens_qids_*.npz")):
            new_name = file.replace("tokens_qids_", "tokens_qids_links_")
            os.rename(file, new_name)


if __name__ == "__main__":
    run_mewsli_mention()
