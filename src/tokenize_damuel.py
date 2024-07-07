from pathlib import Path

from run_action import (
    tokens_for_finetuning_damuel_descriptions,
    tokens_for_finetuning_damuel_links,
)

DAMUEL = Path("/home/farhand/damuel_spark_workdir")
OUT = Path("/home/farhand/tokenized_damuel")

workers = 128
context_size = 64
model_name = "setu4993/LEALLA-base"


def process(p: Path):
    out_lang_path = OUT / p.name
    out_lang_path.mkdir(parents=True, exist_ok=True)
    d = out_lang_path / "descs"
    d.mkdir(parents=True, exist_ok=True)
    l = out_lang_path / "links"
    l.mkdir(parents=True, exist_ok=True)
    tokens_for_finetuning_damuel_descriptions(model_name, p, context_size, d, workers)
    tokens_for_finetuning_damuel_descriptions(model_name, p, context_size, l, workers)


for p in DAMUEL.iterdir():
    if not p.is_dir() or "wikidata" in p.name:
        continue
    process(p)
