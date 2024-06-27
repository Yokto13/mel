#!/bin/bash

set -ueo pipefail

VENV=...
MODEL="setu4993/LEALLA-base"
SOURCE="path/to/mentions.tsv"
EXPECTED_SIZE=64
DEST="..."
WORKERS=1

source $VENV

python run_action.py tokens_mewsli $MODEL $SOURCE $EXPECTED_SIZE $DEST $WORKERS
