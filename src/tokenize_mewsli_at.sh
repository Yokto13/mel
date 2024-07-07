#!/bin/bash

set -ueo pipefail

VENV=../venv/bin/activate
source $VENV
python run_action.py tokens_all_mewsli_at "/home/farhand/bc/data/mewsli/mewsli-9/output/dataset" tokens_mewsli_at 1 64 "setu4993/LEALLA-base"
