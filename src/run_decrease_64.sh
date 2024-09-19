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

LANG="es"
DAMUEL_DESCS_TOKENS_RAW="$OUTPUTS/tokens_damuel_finetuning/es/descs_pages"
# DAMUEL_DESCS_TOKENS_RAW="$OUTPUTS/tokens_damuel_finetuning/es/descs"
# echo "RUNNING WITH DESCS NOT PAGES!!!"
DAMUEL_LINKS_TOKENS_RAW="$OUTPUTS/tokens_damuel_finetuning/es/links"
MEWSLI_TOKENS_RAW="$OUTPUTS/tokens_mewsli_finetuning/$LANG"
MODEL_PATH="/lnet/work/home-students-external/farhan/troja/outputs/models/LEALLA-base"
WORKDIR="$OUTPUTS/workdirs/small_dim_64"
BATCH_SIZE=64
EPOCHS=100
LOGIT_MULTIPLIER=50
# LR=0.00001
LR=0.00001
# TYPE="mentions_gillick_loss"
TYPE="mentions"
N_OF_ROUNDS=8
NEG=7
NEG_SAMPLING_TYPE="top"
TARGET_DIM=64

# copy params
echo "Copying params"
mkdir -p "$WORKDIR"
PARAMS_FILE="$WORKDIR/params.txt"
echo "DAMUEL_DESCS_TOKENS_RAW=$DAMUEL_DESCS_TOKENS_RAW" > "$PARAMS_FILE"
echo "DAMUEL_LINKS_TOKENS_RAW=$DAMUEL_LINKS_TOKENS_RAW" >> "$PARAMS_FILE"
echo "MEWSLI_TOKENS_RAW=$MEWSLI_TOKENS_RAW" >> "$PARAMS_FILE"
echo "MODEL_PATH=$MODEL_PATH" >> "$PARAMS_FILE"
echo "WORKDIR=$WORKDIR" >> "$PARAMS_FILE"
echo "BATCH_SIZE=$BATCH_SIZE" >> "$PARAMS_FILE"
echo "EPOCHS=$EPOCHS" >> "$PARAMS_FILE"
echo "LOGIT_MULTIPLIER=$LOGIT_MULTIPLIER" >> "$PARAMS_FILE"
echo "LR=$LR" >> "$PARAMS_FILE"
echo "TYPE=$TYPE" >> "$PARAMS_FILE"
echo "N_OF_ROUNDS=$N_OF_ROUNDS" >> "$PARAMS_FILE"

create_symlinks() {
    local source_dir="$1"
    local target_dir="$2"
    
    if [ -d "$source_dir" ]; then
        for item in "$source_dir"/*; do
            if [ -e "$item" ]; then
                echo "$item"
                local basename=$(basename "$item")
                local target_path="$target_dir/$basename"
                ln -sf "$item" "$target_path"
            fi
        done
    fi
}


if [ ! -L "$WORKDIR" ]; then
    mkdir -p "$WORKDIR"
fi

DAMUEL_DESCS_TOKENS="$WORKDIR/damuel_descs_together_tokens"
if [ ! -L "$DAMUEL_DESCS_TOKENS" ]; then
    mkdir -p "$DAMUEL_DESCS_TOKENS"
fi

for ((ROUND_ID=0; ROUND_ID<$N_OF_ROUNDS; ROUND_ID++))
do
    DAMUEL_LINKS_TOKENS="$WORKDIR/damuel_links_together_tokens_$ROUND_ID"
    if [ ! -L "$DAMUEL_LINKS_TOKENS" ]; then
        mkdir -p "$DAMUEL_LINKS_TOKENS"
    fi
done

MEWSLI_TOKENS="$WORKDIR/mewsli_together_tokens"
if [ ! -L "$MEWSLI_TOKENS" ]; then
    mkdir -p "$MEWSLI_TOKENS"
fi

create_symlinks $DAMUEL_DESCS_TOKENS_RAW $DAMUEL_DESCS_TOKENS
create_symlinks $MEWSLI_TOKENS_RAW $MEWSLI_TOKENS


if [ ! -e "$WORKDIR/models_0/final.pth" ]; then
    echo "Running round 0"

    ./run_finetuning_round_decrease_dim.sh "$DAMUEL_DESCS_TOKENS_RAW" "$DAMUEL_LINKS_TOKENS_RAW"\
     "$MEWSLI_TOKENS_RAW" "$MODEL_PATH"\
     "$WORKDIR" "$BATCH_SIZE" $(($EPOCHS / 5)) "$LOGIT_MULTIPLIER" "$LR" "None" 0 "$TYPE" "$N_OF_ROUNDS"\
     $NEG 1 $NEG_SAMPLING_TYPE $TARGET_DIM
fi

STATE_DICT="$WORKDIR/models_0/final.pth"

# if [ ! -e "$WORKDIR/models_1/final.pth" ]; then
if [ ! -e "$WORKDIR/models_1/final.pth" ]; then
    echo "Running round 1"

    ./run_finetuning_round_decrease_dim.sh "$DAMUEL_DESCS_TOKENS_RAW" "$DAMUEL_LINKS_TOKENS_RAW" "$MEWSLI_TOKENS_RAW" "$MODEL_PATH"\
     "$WORKDIR" "$BATCH_SIZE" "$EPOCHS" "$LOGIT_MULTIPLIER" "$LR" $STATE_DICT 1 "$TYPE" "$N_OF_ROUNDS" $NEG 1 $NEG_SAMPLING_TYPE $TARGET_DIM
fi

STATE_DICT="$WORKDIR/models_1/final.pth"

if [ ! -e "$WORKDIR/models_2/final.pth" ]; then
    echo "Running round 2"

    ./run_finetuning_round_decrease_dim.sh "$DAMUEL_DESCS_TOKENS_RAW" "$DAMUEL_LINKS_TOKENS_RAW" "$MEWSLI_TOKENS_RAW" "$MODEL_PATH"\
     "$WORKDIR" "$BATCH_SIZE" "$EPOCHS" "$LOGIT_MULTIPLIER" "$LR" $STATE_DICT 2 "$TYPE" "$N_OF_ROUNDS" $NEG 1 $NEG_SAMPLING_TYPE $TARGET_DIM
fi

STATE_DICT="$WORKDIR/models_2/final.pth"

if [ ! -e "$WORKDIR/models_3/final.pth" ]; then
    echo "Running round 3"

    ./run_finetuning_round_decrease_dim.sh "$DAMUEL_DESCS_TOKENS_RAW" "$DAMUEL_LINKS_TOKENS_RAW" "$MEWSLI_TOKENS_RAW" "$MODEL_PATH"\
     "$WORKDIR" "$BATCH_SIZE" "$EPOCHS" "$LOGIT_MULTIPLIER" "$LR" $STATE_DICT 3 "$TYPE" "$N_OF_ROUNDS" $NEG 1 $NEG_SAMPLING_TYPE $TARGET_DIM
fi

STATE_DICT="$WORKDIR/models_3/final.pth"

if [ ! -e "$WORKDIR/models_4/final.pth" ]; then
    echo "Running round 4"

    ./run_finetuning_round_decrease_dim.sh "$DAMUEL_DESCS_TOKENS_RAW" "$DAMUEL_LINKS_TOKENS_RAW" "$MEWSLI_TOKENS_RAW" "$MODEL_PATH"\
     "$WORKDIR" "$BATCH_SIZE" "$EPOCHS" "$LOGIT_MULTIPLIER" "$LR" $STATE_DICT 4 "$TYPE" "$N_OF_ROUNDS" $NEG 1 $NEG_SAMPLING_TYPE $TARGET_DIM
fi

STATE_DICT="$WORKDIR/models_4/final.pth"

if [ ! -e "$WORKDIR/models_5/final.pth" ]; then
    echo "Running round 5"

    ./run_finetuning_round_decrease_dim.sh "$DAMUEL_DESCS_TOKENS_RAW" "$DAMUEL_LINKS_TOKENS_RAW" "$MEWSLI_TOKENS_RAW" "$MODEL_PATH"\
     "$WORKDIR" "$BATCH_SIZE" "$EPOCHS" "$LOGIT_MULTIPLIER" "$LR" $STATE_DICT 5 "$TYPE" "$N_OF_ROUNDS" $NEG 1 $NEG_SAMPLING_TYPE $TARGET_DIM
fi

STATE_DICT="$WORKDIR/models_5/final.pth"

if [ ! -e "$WORKDIR/models_6/final.pth" ]; then
    echo "Running round 6"

    ./run_finetuning_round_decrease_dim.sh "$DAMUEL_DESCS_TOKENS_RAW" "$DAMUEL_LINKS_TOKENS_RAW" "$MEWSLI_TOKENS_RAW" "$MODEL_PATH"\
     "$WORKDIR" "$BATCH_SIZE" "$EPOCHS" "$LOGIT_MULTIPLIER" "$LR" $STATE_DICT 6 "$TYPE" "$N_OF_ROUNDS" $NEG 1 $NEG_SAMPLING_TYPE $TARGET_DIM
fi

STATE_DICT="$WORKDIR/models_6/final.pth"

if [ ! -e "$WORKDIR/models_7/final.pth" ]; then
    echo "Running round 7"

    ./run_finetuning_round_decrease_dim.sh "$DAMUEL_DESCS_TOKENS_RAW" "$DAMUEL_LINKS_TOKENS_RAW" "$MEWSLI_TOKENS_RAW" "$MODEL_PATH"\
     "$WORKDIR" "$BATCH_SIZE" "$EPOCHS" "$LOGIT_MULTIPLIER" "$LR" $STATE_DICT 7 "$TYPE" "$N_OF_ROUNDS" $NEG 1 $NEG_SAMPLING_TYPE $TARGET_DIM
fi
