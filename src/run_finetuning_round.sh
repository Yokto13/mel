#!/bin/bash

set -ueo pipefail

DAMUEL_DESC_TOKENS_RAW=$1
DAMUEL_LINKS_TOKENS_RAW=$2
MEWSLI_TOKENS_RAW=$3
MODEL_PATH=$4
WORKDIR=$5
BATCH_SIZE=$6
EPOCHS=$7
LOGIT_MULTIPLIER=$8
LR=$9
STATE_DICT=${10:-"None"}
echo "STATE_DICT: $STATE_DICT"
ROUND_ID=${11:-"0"}
TYPE=${12}
N_OF_ROUNDS=${13}

DAMUEL_DESC_TOKENS="$WORKDIR/damuel_descs_together_tokens"
if [ ! -L "$DAMUEL_DESC_TOKENS" ]; then
    mkdir -p "$DAMUEL_DESC_TOKENS"
fi

DAMUEL_LINKS_TOKENS="$WORKDIR/damuel_links_together_tokens_$ROUND_ID"
if [ ! -L "$DAMUEL_LINKS_TOKENS" ]; then
    mkdir -p "$DAMUEL_LINKS_TOKENS"
fi

MEWSLI_TOKENS="$WORKDIR/mewsli_together_tokens"
if [ ! -L "$MEWSLI_TOKENS" ]; then
    mkdir -p "$MEWSLI_TOKENS"
fi

POS=1
NEG=${14:-7}
CONTEXT_SIZE=64
STEPS_PER_EPOCH=1000

ACTION_SCRIPT="run_action.py"

ENV="../venv/bin/activate"
source $ENV

# ====================TOKENS COPY====================

# The last two arguments make sure that only part of the tokens is processed
# This ensures that data are split between different rounds
if [ ! "$(ls -A $DAMUEL_LINKS_TOKENS)" ]; then
    python $ACTION_SCRIPT "copy" "$DAMUEL_LINKS_TOKENS_RAW" "$DAMUEL_LINKS_TOKENS" "$N_OF_ROUNDS" "$ROUND_ID"
fi

# ====================DAMUEL DESC EMBS====================

DAMUEL_FOR_INDEX_DIR="$WORKDIR/damuel_for_index_$ROUND_ID"

mkdir -p "$DAMUEL_FOR_INDEX_DIR"

if [ ! "$(ls -A $DAMUEL_FOR_INDEX_DIR)" ]; then
    echo "Running embs generating for damuel"
    python $ACTION_SCRIPT "embs_from_tokens_model_name_and_state_dict" "$DAMUEL_DESC_TOKENS" "$MODEL_PATH" 16384 "$DAMUEL_FOR_INDEX_DIR" "$STATE_DICT"
fi

# ====================DAMUEL DESC TOKEN INDEX====================

INDEX_DIR="$WORKDIR/index_$ROUND_ID"

mkdir -p "$INDEX_DIR"
if [ ! "$(ls -A $INDEX_DIR)" ]; then
    echo "Running token index generating for damuel"
    python $ACTION_SCRIPT "token_index" "$DAMUEL_FOR_INDEX_DIR" "$DAMUEL_DESC_TOKENS" "$INDEX_DIR"
fi

# ====================GENERATING BATCHES====================

BATCH_DIR="$WORKDIR/batches_$ROUND_ID"

mkdir -p "$BATCH_DIR"
if [ ! "$(ls -A $BATCH_DIR)" ]; then
    echo "Running batches generating for damuel"
    echo $ACTION_SCRIPT "generate" "$DAMUEL_LINKS_TOKENS" "$INDEX_DIR" "$BATCH_DIR" "$MODEL_PATH" "$BATCH_SIZE" "$EPOCHS" "$STEPS_PER_EPOCH" "$POS" "$NEG" "$CONTEXT_SIZE" "$TYPE" "$STATE_DICT"
    python $ACTION_SCRIPT "generate" "$DAMUEL_LINKS_TOKENS" "$INDEX_DIR" "$BATCH_DIR" "$MODEL_PATH" "$BATCH_SIZE" "$EPOCHS" "$STEPS_PER_EPOCH" "$POS" "$NEG" "$CONTEXT_SIZE" "$TYPE" "$STATE_DICT"
fi

# ====================TRAINING MODEL====================


MODELS_DIR="$WORKDIR/models_$ROUND_ID"

mkdir -p $MODELS_DIR

if [ ! "$(ls -A $MODELS_DIR)" ]; then
    echo "Running training for damuel"
    python $ACTION_SCRIPT "train" "$BATCH_DIR" "$MODEL_PATH" "$EPOCHS" "$LOGIT_MULTIPLIER" "$LR" "$TYPE" "$MODELS_DIR" "$STATE_DICT"
fi

# ====================EVALUATION====================

DAMUEL_FOR_INDEX_NEW_DIR="$WORKDIR/damuel_for_index_$(($ROUND_ID + 1))"
MEWSLI_EMBS_DIR="$WORKDIR/mewsli_embs_$ROUND_ID"

mkdir -p "$MEWSLI_EMBS_DIR"
mkdir -p "$DAMUEL_FOR_INDEX_NEW_DIR"

python $ACTION_SCRIPT "evaluate" "$DAMUEL_DESC_TOKENS" "$MEWSLI_TOKENS" "$MODEL_PATH"\
 "$DAMUEL_FOR_INDEX_NEW_DIR" "$MEWSLI_EMBS_DIR" "$MODELS_DIR/final.pth"


# ====================CLEAN UP====================

rm -r "$DAMUEL_FOR_INDEX_DIR"
rm -r "$INDEX_DIR"
rm -r "$MEWSLI_EMBS_DIR"
rm -r "$BATCH_DIR"
