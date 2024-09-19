#!/bin/bash

set -ueo pipefail

cd ../../

VENV=../venv/bin/activate
source $VENV

INPUT="$1"
OUTPUT_TYPE="$2"
RESULT="$3"

# RESULT="$OUTPUTS/embs_damuel_at/$LANG"
DESCS="$RESULT/descs"
DESCS_PAGES="$RESULT/descs_pages"
LINKS="$RESULT/links"
# INPUT="$OUTPUTS/tokens_damuel_at/$LANG"

if [ -d "$DESCS" ]; then
    echo "Directory exists. Cleaning it."
    rm -r "$DESCS"
    mkdir $DESCS
fi

mkdir -p $DESCS
python run_action.py "embs_from_tokens_and_model_name_at" "$INPUT/descs" "setu4993/LEALLA-small" 196608 "$DESCS" "$OUTPUT_TYPE"

if [ -d "$LINKS" ]; then
    echo "Directory exists. Cleaning it."
    rm -r "$LINKS"
    mkdir $LINKS
fi
echo "Directory does not exist. Creating $LINKS."
mkdir -p $LINKS
python run_action.py "embs_from_tokens_and_model_name_at" "$INPUT/links" "setu4993/LEALLA-small" 196608 "$LINKS" "$OUTPUT_TYPE"

# python run_action.py "embs_from_tokens_and_model_name_at" "$OUTPUTS/tokens_damuel_at_pages/$LANG/descs" "setu4993/LEALLA-small" 196608 "$DESCS_PAGES"
