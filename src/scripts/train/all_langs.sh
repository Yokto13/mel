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
GENERAL_CONFIG_PATH="../configs/general.gin"
MODEL_CONFIG_PATH="../configs/lealla.gin"
TRAIN_CONFIG_PATH="../configs/train.gin"

DAMUEL_DESCS_TOKENS_RAW="$OUTPUTS/all/descs_pages"
DAMUEL_LINKS_TOKENS_RAW="$OUTPUTS/all/links"
MEWSLI_TOKENS_RAW="$OUTPUTS/tokens_mewsli_finetuning"
WORKDIR="$OUTPUTS/workdirs/all_new"
N_OF_ROUNDS=10

# copy params
echo "Copying params"
mkdir -p "$WORKDIR"
PARAMS_FILE="$WORKDIR/params.txt"
echo "DAMUEL_DESCS_TOKENS_RAW=$DAMUEL_DESCS_TOKENS_RAW" > "$PARAMS_FILE"
echo "DAMUEL_LINKS_TOKENS_RAW=$DAMUEL_LINKS_TOKENS_RAW" >> "$PARAMS_FILE"
echo "MEWSLI_TOKENS_RAW=$MEWSLI_TOKENS_RAW" >> "$PARAMS_FILE"
echo "WORKDIR=$WORKDIR" >> "$PARAMS_FILE"
echo "N_OF_ROUNDS=$N_OF_ROUNDS" >> "$PARAMS_FILE"

run_ml_finetuning_round() {
    local DAMUEL_DESCS_TOKENS_RAW=$1
    local DAMUEL_LINKS_TOKENS_RAW=$2
    local MEWSLI_TOKENS_RAW=$3
    local WORKDIR=$4
    local STATE_DICT=${5:-"None"}
    local ROUND_ID=${6:-"0"}
    local N_OF_ROUNDS=${7}
    local CARDS=${8:-8}

    local STEPS_PER_EPOCH=1000
    local POS=1

    # Multiple by 2 to make sure that if a link contained something faulty we can skip it.
    local LINKS_PER_ROUND=$(($STEPS_PER_EPOCH * 10))
    echo "LPR $LINKS_PER_ROUND"

    local ACTION_SCRIPT="run_action_gin.py $GENERAL_CONFIG_PATH $MODEL_CONFIG_PATH $TRAIN_CONFIG_PATH"

    ENV="../venv/bin/activate"
    source $ENV

    # ====================TOKENS COPY====================

    # The last two arguments make sure that only part of the tokens is processed
    # This ensures that data are split between different rounds
    local DAMUEL_LINKS_TOKENS="$WORKDIR/damuel_links_together_tokens_$ROUND_ID"
    if [ ! "$(ls -A $DAMUEL_LINKS_TOKENS)" ]; then
        python $ACTION_SCRIPT "copy" \
            --source="$DAMUEL_LINKS_TOKENS_RAW" \
            --dest="$DAMUEL_LINKS_TOKENS" \
            --m="$N_OF_ROUNDS" \
            --r="$ROUND_ID" \
            --max_to_copy="$LINKS_PER_ROUND"
    fi

    # ====================DAMUEL DESC EMBS====================

    local DAMUEL_FOR_INDEX_DIR="$WORKDIR/damuel_for_index_$ROUND_ID"

    mkdir -p "$DAMUEL_FOR_INDEX_DIR"

    if [ ! "$(ls -A $DAMUEL_FOR_INDEX_DIR)" ]; then
        echo "Running embs generating for damuel"
        sbatch --wait -p "gpu-troja,gpu-ms" -G 8 -C "gpuram24G|gpuram16G" --nodes=1 --mem=150G run ../venv/bin/python $ACTION_SCRIPT "embs_from_tokens_model_name_and_state_dict" \
            --source_path="$DAMUEL_DESCS_TOKENS_RAW" \
            --dest_path="$DAMUEL_FOR_INDEX_DIR" \
            --state_dict_path="$STATE_DICT"
    fi

    # ====================DAMUEL LINKS EMBEDDING====================

    # for searcher we need to embed links so we can construct batches

    local DAMUEL_LINKS_DIR="$WORKDIR/links_embs_$ROUND_ID"

    mkdir -p "$DAMUEL_LINKS_DIR"

    if [ ! "$(ls -A $DAMUEL_LINKS_DIR)" ]; then
        echo "Running embs generating for damuel links"
        sbatch --wait -p "gpu-troja,gpu-ms" -G 8 -C "gpuram24G|gpuram16G" --nodes=1 --mem=150G run ../venv/bin/python $ACTION_SCRIPT "embed_links_for_generation" \
            --source_path="$DAMUEL_LINKS_TOKENS" \
            --dest_path="$DAMUEL_LINKS_DIR" \
            --state_dict_path="$STATE_DICT"
    fi

    # ====================GENERATING BATCHES====================

    local BATCH_DIR="$WORKDIR/batches_$ROUND_ID"

    mkdir -p "$BATCH_DIR"
    if [ ! "$(ls -A $BATCH_DIR)" ]; then
        echo "Running batches generating for damuel"
        echo $ACTION_SCRIPT "generate" \
            --LINKS_EMBS_DIR="$DAMUEL_LINKS_DIR" \
            --INDEX_TOKENS_DIR="$DAMUEL_DESCS_TOKENS_RAW" \
            --INDEX_EMBS_QIDS_DIR="$DAMUEL_FOR_INDEX_DIR" \
            --OUTPUT_DIR="$BATCH_DIR"
        sbatch --wait -p  "gpu-troja,gpu-ms" -N 1 -G 2 -C "gpuram40G|gpuram48G" --mem=150G  run ../venv/bin/python $ACTION_SCRIPT "generate" \
            --LINKS_EMBS_DIR="$DAMUEL_LINKS_DIR" \
            --INDEX_TOKENS_DIR="$DAMUEL_DESCS_TOKENS_RAW" \
            --INDEX_EMBS_QIDS_DIR="$DAMUEL_FOR_INDEX_DIR" \
            --OUTPUT_DIR="$BATCH_DIR" \
            --GENERATE_Y=False
    fi

    # ====================TRAINING MODEL====================

    local MODELS_DIR="$WORKDIR/models_$ROUND_ID"

    mkdir -p $MODELS_DIR

    if [ ! "$(ls -A $MODELS_DIR)" ]; then
        echo "Running training for damuel"
        echo $ACTION_SCRIPT "train_ddp" \
            --DATASET_DIR="$BATCH_DIR" \
            --MODEL_SAVE_DIR="$MODELS_DIR" \
            --STATE_DICT_PATH="$STATE_DICT"
        sbatch --wait -p "gpu-troja,gpu-ms" -G $CARDS -N1 -C "gpuram24G|gpuram40G" --mem=150G run ../venv/bin/python $ACTION_SCRIPT "train_ddp" \
            --DATASET_DIR="$BATCH_DIR" \
            --MODEL_SAVE_DIR="$MODELS_DIR" \
            --STATE_DICT_PATH="$STATE_DICT"
    fi

    # ====================EVALUATION====================

    local NEXT_INDEX=$(($ROUND_ID + 1))
    local DAMUEL_FOR_INDEX_NEW_DIR="$WORKDIR/damuel_for_index_$NEXT_INDEX"
    mkdir -p "$DAMUEL_FOR_INDEX_NEW_DIR"

    if [ ! "$(ls -A $DAMUEL_FOR_INDEX_NEW_DIR)" ]; then
        echo "Running embs generating for damuel"
        sbatch --wait -p "gpu-troja,gpu-ms" -G 8 -C "gpuram24G|gpuram16G" --nodes=1 --mem=150G run ../venv/bin/python $ACTION_SCRIPT "embs_from_tokens_model_name_and_state_dict" \
            --source_path="$DAMUEL_DESCS_TOKENS_RAW" \
            --dest_path="$DAMUEL_FOR_INDEX_NEW_DIR" \
            --state_dict_path="$MODELS_DIR/final.pth"
    fi

    local LANGUAGES=("ar" "de" "en" "es" "ja" "fa" "sr" "ta" "tr")

    # Parent directory for all language-specific token directories
    local PARENT_TOKEN_DIR="$WORKDIR/mewsli_tokens_raw"

    for LANG in "${LANGUAGES[@]}"; do
        echo "Processing language: $LANG"
        
        # Set language-specific paths
        local LANG_TOKEN_DIR="$MEWSLI_TOKENS_RAW/$LANG"
        local MEWSLI_EMBS_DIR="$WORKDIR/mewsli_embs_${LANG}_$ROUND_ID"
        
        mkdir -p "$MEWSLI_EMBS_DIR"
        
        if [ ! "$(ls -A $MEWSLI_EMBS_DIR)" ]; then
            echo "Running embs generating for mewsli - Language: $LANG"
            sbatch --wait -p "gpu-troja,gpu-ms" -G 1 -C "gpuram24G" --mem=50G run ../venv/bin/python $ACTION_SCRIPT "embs_from_tokens_model_name_and_state_dict" \
                --source_path="$LANG_TOKEN_DIR" \
                --dest_path="$MEWSLI_EMBS_DIR" \
                --state_dict_path="$MODELS_DIR/final.pth"
        fi
        
        sbatch -p "gpu-troja,gpu-ms" -G 1 -C "gpuram24G|gpuram40G|gpuram48G" --mem=50G run ../venv/bin/python $ACTION_SCRIPT "recalls" \
            --damuel_dir="$DAMUEL_FOR_INDEX_NEW_DIR" \
            --mewsli_dir="$MEWSLI_EMBS_DIR"
        
        echo "Completed processing for language: $LANG"
        echo "----------------------------------------"
    done
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

STATE_DICT="None"

for ((ROUND_ID=0; ROUND_ID<$N_OF_ROUNDS; ROUND_ID++))
do
    if [ ! -e "$WORKDIR/models_$ROUND_ID/final.pth" ]; then
        echo "Running round $ROUND_ID"

        run_ml_finetuning_round "$DAMUEL_DESCS_TOKENS_RAW" "$DAMUEL_LINKS_TOKENS_RAW" \
            "$MEWSLI_TOKENS_RAW" \
            "$WORKDIR" "$STATE_DICT" \
            "$ROUND_ID" "$N_OF_ROUNDS" 8
    fi

    STATE_DICT="$WORKDIR/models_$ROUND_ID/final.pth"
done