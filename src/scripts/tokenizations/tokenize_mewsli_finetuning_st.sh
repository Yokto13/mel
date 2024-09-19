#!/bin/bash

set -ueo pipefail

cd ../../

VENV=../venv/bin/activate
source $VENV

RESULT="$OUTPUTS/tokens_mewsli_finetuning_st"

if [ -d "$RESULT" ]; then
    echo "Directory exists. Removing it."
    rm -r "$RESULT"
else
    echo "Directory does not exist. Creating it..."
    mkdir -p "$RESULT"
fi

python run_action.py tokens_for_all_mewsli_finetuning "$MEWSLI/mewsli-9/output/dataset" $RESULT 1 64 "/lnet/work/home-students-external/farhan/troja/outputs/models/multi-qa-MiniLM-L6-cos-v1"
