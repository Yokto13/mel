#!/bin/bash

set -ueo pipefail

VENV=../venv/bin/activate
MODEL="setu4993/LEALLA-base"
SOURCE="/home/farhand/damuel_spark_workdir/damuel_1.0_es"
EXPECTED_SIZE=64
DEST="/home/farhand/dump"
WORKERS=10

source $VENV

python run_action.py tokens_links_at $MODEL $SOURCE $EXPECTED_SIZE $DEST $WORKERS
