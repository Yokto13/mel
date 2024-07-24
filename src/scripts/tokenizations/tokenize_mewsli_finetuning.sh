#!/bin/bash

set -ueo pipefail

cd ../../

VENV=../venv/bin/activate
source $VENV
python run_action.py tokens_for_all_mewsli_finetuning "/home/farhand/bc/data/mewsli/mewsli-9/output/dataset" "/home/farhand/tokens_mewsli_finetuning" 1 64 "setu4993/LEALLA-base"
