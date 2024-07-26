#!/bin/bash

set -ueo pipefail

cd ../../

VENV=../venv/bin/activate
source $VENV

RESULT="$OUTPUTS/tokens_damuel_finetuning"

if [ -d "$RESULT" ]; then
    echo "Directory exists. Removing it."
    rm -r "$RESULT"
else
    echo "Directory does not exist. Creating it."
    mkdir -p "$RESULT"
fi

python run_action.py tokens_for_all_damuel_finetuning "$DAMUEL" "$RESULT" 128 64 "setu4993/LEALLA-base"

