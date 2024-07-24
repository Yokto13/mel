#!/bin/bash

set -ueo pipefail

cd ../../

VENV=../venv/bin/activate
source $VENV

RESULT="$OUTPUTS/tokens_damuel_at"

if [ -d "$RESULT" ]; then
    echo "Directory exists. Removing it."
    rm -r "$RESULT"
else
    echo "Directory does not exist."
fi

python run_action.py tokens_for_all_damuel_at "$DAMUEL/1.0" "$RESULT" 128 64 "setu4993/LEALLA-base"
