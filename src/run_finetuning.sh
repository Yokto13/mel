#!/bin/bash

# Runs the complete finetuning process.
# Expects tokens to be in the dirs specified below.
# Additionaly, one can specify additional parameters.
# For running, please also set up/fix the path to venv in run_finetuning_action.sh

set -ueo pipefail

# What we need to do
# Generate embs for index
# Build token index
# Generate batches
# Train model for the first time
# Run evaluation
#   generate embs from the new model damuel desc
#   generate embs from the new model mewsli
#   evaluate
# Build token index
# Generate batches
# Train model for the second time
# Run evaluation
#   generate embs from the new model damuel desc
#   generate embs from the new model mewsli
#   evaluate
# ...

DAMUEL_DESC_TOKENS="/home/farhand/bc_jobs_important/desc_small"
DAMUEL_LINKS_TOKENS="/home/farhand/bc_jobs_important/links_small"
MEWSLI_TOKENS="/home/farhand/bc_jobs_important/mewsli_together_tokens"
MODEL_PATH="setu4993/LEALLA-base"
WORKDIR="test_finetuning"
BATCH_SIZE=32
EPOCHS=10
LOGIT_MULTIPLIER=10
# LR=0.00001
LR=0.00001
# TYPE="mentions_gillick_loss"
TYPE="mentions"
N_OF_ROUNDS=2
NEG=7

# copy params
echo "Copying params"
mkdir -p "$WORKDIR"
PARAMS_FILE="$WORKDIR/params.txt"
echo "DAMUEL_DESC_TOKENS=$DAMUEL_DESC_TOKENS" > "$PARAMS_FILE"
echo "DAMUEL_LINKS_TOKENS=$DAMUEL_LINKS_TOKENS" >> "$PARAMS_FILE"
echo "MEWSLI_TOKENS=$MEWSLI_TOKENS" >> "$PARAMS_FILE"
echo "MODEL_PATH=$MODEL_PATH" >> "$PARAMS_FILE"
echo "WORKDIR=$WORKDIR" >> "$PARAMS_FILE"
echo "BATCH_SIZE=$BATCH_SIZE" >> "$PARAMS_FILE"
echo "EPOCHS=$EPOCHS" >> "$PARAMS_FILE"
echo "LOGIT_MULTIPLIER=$LOGIT_MULTIPLIER" >> "$PARAMS_FILE"
echo "LR=$LR" >> "$PARAMS_FILE"
echo "TYPE=$TYPE" >> "$PARAMS_FILE"
echo "N_OF_ROUNDS=$N_OF_ROUNDS" >> "$PARAMS_FILE"


if [ ! -e "$WORKDIR/models_0/final.pth" ]; then
    echo "Running round 0"

    ./run_finetuning_round.sh "$DAMUEL_DESC_TOKENS" "$DAMUEL_LINKS_TOKENS"\
     "$MEWSLI_TOKENS" "$MODEL_PATH"\
     "$WORKDIR" "$BATCH_SIZE" $(($EPOCHS / 5)) "$LOGIT_MULTIPLIER" "$LR" "None" 0 "$TYPE" "$N_OF_ROUNDS"\
     $NEG
fi

STATE_DICT="$WORKDIR/models_0/final.pth"

if [ ! -e "$WORKDIR/models_1/final.pth" ]; then
    echo "Running round 1"

    ./run_finetuning_round.sh "$DAMUEL_DESC_TOKENS" "$DAMUEL_LINKS_TOKENS" "$MEWSLI_TOKENS" "$MODEL_PATH"\
     "$WORKDIR" "$BATCH_SIZE" "$EPOCHS" "$LOGIT_MULTIPLIER" "$LR" $STATE_DICT 1 "$TYPE" "$N_OF_ROUNDS" $NEG
fi
