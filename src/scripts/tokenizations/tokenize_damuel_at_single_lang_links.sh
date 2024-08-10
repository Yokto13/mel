#!/bin/bash

set -ueo pipefail

cd ../../

VENV=../venv/bin/activate
source $VENV

LANG="$1"

RESULT="$OUTPUTS/tmp/tokens_damuel_at_base/$LANG"
DESCS="$RESULT/descs"
LINKS="$RESULT/links"

if [ -d "$LINKS" ]; then
    echo "Directory exists. Removing it."
    rm -r "$LINKS"
    mkdir -p "$LINKS"
else
    echo "Directory does not exist. Creating it."
    mkdir -p "$LINKS"
fi

python run_action.py tokens_links_at "setu4993/LEALLA-base" "$DAMUEL/damuel_1.0_$LANG" 64 "$LINKS" 128
