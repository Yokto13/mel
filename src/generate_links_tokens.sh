#!/bin/bash

set -ueo pipefail

VENV=...
MODEL="setu4993/LEALLA-base"
SOURCE="path/to/damuel"
EXPECTED_SIZE=64
DEST="..."
WORKERS=1

source $VENV

python run_action.py tokens_links $MODEL $SOURCE $EXPECTED_SIZE $DEST $WORKERS
