#!/bin/bash

set -ueo pipefail

DAMUEL_DESC_TOKENS=$1
MEWSLI_TOKENS=$2
MODEL_PATH=$3
WORKDIR=$4
STATE_DICT=$5

ACTION_SCRIPT="run_action.py"

DAMUEL_DIR="$WORKDIR/damuel"
MEWSLI_DIR="$WORKDIR/mewsli"

DAMUEL_EMBS_DIR="$WORKDIR/damuel_embs"
MEWSLI_EMBS_DIR="$WORKDIR/mewsli_embs"

mkdir -p "$DAMUEL_DIR"
mkdir -p "$MEWSLI_DIR"

mkdir -p $DAMUEL_EMBS_DIR
mkdir -p $MEWSLI_EMBS_DIR

ENV="/home/farhand/bc/venv/bin/activate"
source $ENV

if [ ! "$(ls -A $DAMUEL_DIR)" ]; then
    python $ACTION_SCRIPT "copy" "$DAMUEL_DESC_TOKENS" "$DAMUEL_DIR"
fi

if [ ! "$(ls -A $MEWSLI_DIR)" ]; then
    python $ACTION_SCRIPT "copy" "$MEWSLI_TOKENS" "$MEWSLI_DIR"
fi

# Hack around the fact that evaluate.py expects files to be start with mentions.
# This is due to the fact that evaluate is part of the finetuning loop.
# Rename does nothing when the files are already named correctly.
# Needed for compatibility issues with older tokenizers.
python $ACTION_SCRIPT "rename" "$DAMUEL_DIR" "entity_names" "mentions"
python $ACTION_SCRIPT "rename" "$MEWSLI_DIR" "entity_names" "mentions"

python $ACTION_SCRIPT "evaluate" "$DAMUEL_DIR" "$MEWSLI_DIR" "$MODEL_PATH"\
 "$DAMUEL_EMBS_DIR" "$MEWSLI_EMBS_DIR" "$STATE_DICT"

