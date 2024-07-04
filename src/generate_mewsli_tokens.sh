#!/bin/bash

set -ueo pipefail

VENV=../venv/bin/activate
MODEL="setu4993/LEALLA-base"
SOURCE="/home/farhand/bc/data/mewsli/mewsli-9/output/dataset/ta/mentions.tsv"
EXPECTED_SIZE=64
DEST="/home/farhand/dump"
WORKERS=1

source $VENV

python run_action.py tokens_mewsli $MODEL $SOURCE $EXPECTED_SIZE $DEST $WORKERS
