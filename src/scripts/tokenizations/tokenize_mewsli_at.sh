#!/bin/bash

set -ueo pipefail

cd ../../

VENV=../venv/bin/activate
source $VENV

RESULT="$OUTPUTS/tokens_mewsli_at"

if [ -d "$RESULT" ]; then
    echo "Directory exists. Removing it."
    rm -r "$RESULT"
else
    echo "Directory does not exist."
fi

python run_action.py tokens_for_all_mewsli_at "$MEWSLI/mewsli-9/output/dataset" $RESULT 1 64 "setu4993/LEALLA-base"
