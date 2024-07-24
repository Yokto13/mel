#!/bin/bash

set -ueo pipefail

source ../venv/bin/activate

python run_action.py "embs_from_tokens_and_model_name" "$DAMUEL/1.0/damuel_1.0_ja/links" "setu4993/LEALLA-base" 4096 "$OUTPUTS/olpeat_data/ja_dam"
