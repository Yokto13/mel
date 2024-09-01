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
CARDS=${15:-8}
NEG_SAMPLING_TYPE=${16:-"top"}
CONTEXT_SIZE=64
STEPS_PER_EPOCH=1000

# Multiple by 2 to make sure that if a link contained something faulty we can skip it.
LINKS_PER_ROUND=$(($STEPS_PER_EPOCH * $BATCH_SIZE * $EPOCHS * 4))
echo "LPR $LINKS_PER_ROUND"

ACTION_SCRIPT="run_action.py"

ENV="../venv/bin/activate"
source $ENV

# ====================TOKENS COPY====================

# The last two arguments make sure that only part of the tokens is processed
# This ensures that data are split between different rounds
if [ ! "$(ls -A $DAMUEL_LINKS_TOKENS)" ]; then
    python $ACTION_SCRIPT "copy" "$DAMUEL_LINKS_TOKENS_RAW" "$DAMUEL_LINKS_TOKENS" "$N_OF_ROUNDS" "$ROUND_ID" "$LINKS_PER_ROUND"
fi

# ====================DAMUEL DESC EMBS====================

DAMUEL_FOR_INDEX_DIR="$WORKDIR/damuel_for_index_$ROUND_ID"

mkdir -p "$DAMUEL_FOR_INDEX_DIR"

if [ ! "$(ls -A $DAMUEL_FOR_INDEX_DIR)" ]; then
    echo "Running embs generating for damuel"
    sbatch --wait -p "gpu-troja,gpu-ms" -G 8 -C "gpuram24G" --mem=150G run ../venv/bin/python $ACTION_SCRIPT "embs_from_tokens_model_name_and_state_dict" "$DAMUEL_DESC_TOKENS" "$MODEL_PATH" 130000 "$DAMUEL_FOR_INDEX_DIR" "$STATE_DICT"
fi

# ====================DAMUEL LINKS EMBEDDING====================

# for searcher we need to embed links so we can construct batches

# INDEX_DIR="$WORKDIR/index_$ROUND_ID"

DAMUEL_LINKS_DIR="$WORKDIR/links_embs_$ROUND_ID"

mkdir -p "$DAMUEL_LINKS_DIR"

if [ ! "$(ls -A $DAMUEL_LINKS_DIR)" ]; then
    echo "Running embs generating for damuel links"
    sbatch --wait -p "gpu-troja,gpu-ms" -G 8 -C "gpuram24G" --mem=150G run ../venv/bin/python $ACTION_SCRIPT "embed_links_for_generation" "$DAMUEL_LINKS_TOKENS" "$MODEL_PATH" 130000 "$DAMUEL_LINKS_DIR" "$STATE_DICT"
fi

# ====================GENERATING BATCHES====================

BATCH_DIR="$WORKDIR/batches_$ROUND_ID"

mkdir -p "$BATCH_DIR"
if [ ! "$(ls -A $BATCH_DIR)" ]; then
    echo "Running batches generating for damuel"
    echo $ACTION_SCRIPT "generate" "$DAMUEL_LINKS_DIR" "$DAMUEL_DESC_TOKENS" "$DAMUEL_FOR_INDEX_DIR" "$BATCH_DIR" "$BATCH_SIZE" "$EPOCHS" "$STEPS_PER_EPOCH" "$NEG" "$CONTEXT_SIZE" $NEG_SAMPLING_TYPE
    # sbatch --wait -p "cpu-ms,cpu-troja" -c60 --mem=100G --exclude="belzebub,iridium" run ../venv/bin/python $ACTION_SCRIPT "generate" "$DAMUEL_LINKS_DIR" "$DAMUEL_DESC_TOKENS" "$DAMUEL_FOR_INDEX_DIR" "$BATCH_DIR" "$BATCH_SIZE" "$EPOCHS" "$STEPS_PER_EPOCH" "$NEG" "$CONTEXT_SIZE"
    sbatch --wait -p  "gpu-troja,gpu-ms" -G 1 -C "gpuram40G" --mem=70G  run ../venv/bin/python $ACTION_SCRIPT "generate" "$DAMUEL_LINKS_DIR" "$DAMUEL_DESC_TOKENS" "$DAMUEL_FOR_INDEX_DIR" "$BATCH_DIR" "$BATCH_SIZE" "$EPOCHS" "$STEPS_PER_EPOCH" "$NEG" "$CONTEXT_SIZE" $NEG_SAMPLING_TYPE
    # python $ACTION_SCRIPT "generate" "$DAMUEL_LINKS_DIR" "$DAMUEL_DESC_TOKENS" "$DAMUEL_FOR_INDEX_DIR" "$BATCH_DIR" "$MODEL_PATH" "$BATCH_SIZE" "$EPOCHS" "$STEPS_PER_EPOCH" "$NEG" "$CONTEXT_SIZE" "$STATE_DICT"
fi

# ====================TRAINING MODEL====================


MODELS_DIR="$WORKDIR/models_$ROUND_ID"

mkdir -p $MODELS_DIR

if [ ! "$(ls -A $MODELS_DIR)" ]; then
    echo "Running training for damuel"
    echo $ACTION_SCRIPT "train" "$BATCH_DIR" "$MODEL_PATH" "$EPOCHS" "$LOGIT_MULTIPLIER" "$LR" "$TYPE" "$MODELS_DIR" "$STATE_DICT"
    # sbatch --wait -p "gpu-troja,gpu-ms" -G $CARDS -C "gpuram24G|gpuram40G" --mem=150G run ../venv/bin/python $ACTION_SCRIPT "train" "$BATCH_DIR" "$MODEL_PATH" "$EPOCHS" "$LOGIT_MULTIPLIER" "$LR" "$TYPE" "$MODELS_DIR" "$STATE_DICT"
    # sbatch --wait -p "gpu-troja,gpu-ms" -G $CARDS -C "gpuram24G|gpuram40G" --mem=250G run ../venv/bin/python $ACTION_SCRIPT "train_ddp" "$BATCH_DIR" "$MODEL_PATH" "$EPOCHS" "$LOGIT_MULTIPLIER" "$LR" "$TYPE" "$MODELS_DIR" "$STATE_DICT"
    sbatch --wait -p "gpu-troja" -G $CARDS -C "gpuram40G|gpuram24G" --mem=200G run ../venv/bin/python $ACTION_SCRIPT "train_ddp" "$BATCH_DIR" "$MODEL_PATH" "$EPOCHS" "$LOGIT_MULTIPLIER" "$LR" "$TYPE" "$MODELS_DIR" "$STATE_DICT" True
fi

# ====================EVALUATION====================

NEXT_INDEX=$(($ROUND_ID + 1))
DAMUEL_FOR_INDEX_NEW_DIR="$WORKDIR/damuel_for_index_$NEXT_INDEX"
MEWSLI_EMBS_DIR="$WORKDIR/mewsli_embs_$ROUND_ID"

mkdir -p "$MEWSLI_EMBS_DIR"
mkdir -p "$DAMUEL_FOR_INDEX_NEW_DIR"

if [ ! "$(ls -A $MEWSLI_EMBS_DIR)" ]; then
    echo "Running embs generating for mewsli"
    sbatch --wait -p "gpu-troja,gpu-ms" -G 1 -C "gpuram24G" --mem=50G run ../venv/bin/python $ACTION_SCRIPT "embs_from_tokens_model_name_and_state_dict" "$MEWSLI_TOKENS_RAW" "$MODEL_PATH" 16384 "$MEWSLI_EMBS_DIR" "$MODELS_DIR/final.pth"
fi

if [ ! "$(ls -A $DAMUEL_FOR_INDEX_NEW_DIR)" ]; then
    echo "Running embs generating for damuel"
    sbatch --wait -p "gpu-troja,gpu-ms" -G 8 -C "gpuram24G" --mem=150G run ../venv/bin/python $ACTION_SCRIPT "embs_from_tokens_model_name_and_state_dict" "$DAMUEL_DESC_TOKENS" "$MODEL_PATH" 130000 "$DAMUEL_FOR_INDEX_NEW_DIR" "$MODELS_DIR/final.pth"
    # ../venv/bin/python $ACTION_SCRIPT "embs_from_tokens_model_name_and_state_dict" "$DAMUEL_DESC_TOKENS" "$MODEL_PATH" 130000 "$DAMUEL_FOR_INDEX_NEW_DIR" "$MODELS_DIR/final.pth"
fi

# sbatch --wait -p "cpu-ms,cpu-troja" -c10 --mem=100G --exclude="belzebub,iridium" run ../venv/bin/python $ACTION_SCRIPT "recalls" "$DAMUEL_FOR_INDEX_NEW_DIR" "$MEWSLI_EMBS_DIR"
sbatch -p "cpu-ms,cpu-troja" -c60 --mem=100G --exclude="belzebub,iridium" run ../venv/bin/python $ACTION_SCRIPT "recalls" "$DAMUEL_FOR_INDEX_NEW_DIR" "$MEWSLI_EMBS_DIR"

# ====================CLEAN UP====================

# rm -r "$DAMUEL_FOR_INDEX_DIR"
# rm -r "$MEWSLI_EMBS_DIR"
# rm -r "$BATCH_DIR"
# rm -r "$DAMUEL_LINKS_DIR"
