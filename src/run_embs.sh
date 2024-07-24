#!/bin/bash

set -ueo pipefail

source ../venv/bin/activate

python run_action.py "embs_from_tokens" "/home/farhand/tokens_damuel_at/damuel_1.0_ja/links" "setu4993/LEALLA-base" 65536 "/home/farhand/mel-reborn/olpeat_data/ja_dam"
