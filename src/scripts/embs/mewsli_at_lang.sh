#!/bin/bash

set -ueo pipefail

cd ../../

VENV=../venv/bin/activate
source $VENV

LANG="$1"

RESULT="$OUTPUTS/embs_mewsli_at/$LANG"
INPUT="$OUTPUTS/tokens_mewsli_at/$LANG"

if [ -d "$RESULT" ]; then
    echo $RESULT
    echo "Directory exists. Cleaning it."
    rm -r "$RESULT"
    mkdir $RESULT
else
    echo "Directory does not exist. Creating $RESULT."
    mkdir $RESULT
fi

python run_action.py "embs_from_tokens_and_model_name" $INPUT "setu4993/LEALLA-base" 16384 $RESULT