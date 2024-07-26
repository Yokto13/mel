#!/bin/bash

set -ueo pipefail

cd ../../

VENV=../venv/bin/activate
source $VENV

RESULT="$OUTPUTS/tokens_damuel_at_pages"

if [ -d "$RESULT" ]; then
    echo "Directory exists. Removing it."
    rm -r "$RESULT"
else
    echo "Directory does not exist."
fi

python run_action.py tokens_for_all_damuel_at_pages "$DAMUEL" "$RESULT" 100 64 "setu4993/LEALLA-base"
