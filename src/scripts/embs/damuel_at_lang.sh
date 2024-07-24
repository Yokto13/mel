#!/bin/bash

set -ueo pipefail

cd ../../

VENV=../venv/bin/activate
source $VENV

LANG="$1"

RESULT="$OUTPUTS/embs_damuel_at/$LANG"
DESCS="$RESULT/descs"
LINKS="$RESULT/links"
INPUT="$OUTPUTS/tokens_damuel_at/$LANG"

if [ -d "$DESCS" ]; then
    echo "Directory exists. Cleaning it."
    rm -r "$DESCS"
    mkdir $DESCS
else
    echo "Directory does not exist. Creating $DESCS."
    mkdir -p $DESCS
fi

if [ -d "$LINKS" ]; then
    echo "Directory exists. Cleaning it."
    rm -r "$LINKS"
    mkdir $LINKS
else
    echo "Directory does not exist. Creating $LINKS."
    mkdir -p $LINKS
fi

python run_action.py "embs_from_tokens_and_model_name" "$INPUT/descs" "setu4993/LEALLA-base" 131072 $DESCS
python run_action.py "embs_from_tokens_and_model_name" "$INPUT/links" "setu4993/LEALLA-base" 131072 $LINKS