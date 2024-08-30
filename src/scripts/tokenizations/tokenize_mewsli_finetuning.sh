#!/bin/bash

set -ueo pipefail

cd ../../

VENV=../venv/bin/activate
source $VENV

RESULT="$OUTPUTS/tokens_mewsli_finetuning_128"

if [ -d "$RESULT" ]; then
    echo "Directory exists. Removing it."
    rm -r "$RESULT"
else
    echo "Directory does not exist. Creating it..."
    mkdir -p "$RESULT"
fi

python run_action.py tokens_for_all_mewsli_finetuning "$MEWSLI/mewsli-9/output/dataset" $RESULT 1 128 "/lnet/work/home-students-external/farhan/troja/outputs/models/LEALLA-base"
