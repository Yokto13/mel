from pathlib import Path

from run_action import (
    tokens_for_finetuning_mewsli,
)

MEWSLI = Path("/home/farhand/bc/data/mewsli/mewsli-9/output/dataset")
OUT = Path("/home/farhand/tokenized_mewsli")

workers = 128
context_size = 64
model_name = "setu4993/LEALLA-base"


def process(p: Path):
    out_lang_path = OUT / p.name
    out_lang_path.mkdir(parents=True, exist_ok=True)
    tokens_for_finetuning_mewsli(model_name, p, context_size, out_lang_path, workers)


for p in MEWSLI.iterdir():
    if not p.is_dir() or "wikidata" in p.name:
        continue
    process(p)
