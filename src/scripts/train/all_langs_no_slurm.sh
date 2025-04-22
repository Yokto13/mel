#!/bin/bash

# Runs the complete finetuning process.
# Expects tokens to be in the dirs specified below.
# Additionaly, one can specify additional parameters.
# For running, please also set up/fix the path to venv in run_finetuning_action.sh

set -ueo pipefail

cd ../../

echo "Running all_langs.sh"
echo "Current directory: $(pwd)"

MODEL_CONFIG_PATH="../configs/lealla_m.gin"
TRAIN_CONFIG_PATH="../configs/train.gin"

DAMUEL_DESCS_TOKENS_RAW="$OUTPUTS/v2/descs_pages"
DAMUEL_LINKS_TOKENS_RAW="$OUTPUTS/v2/links"
MEWSLI_TOKENS_RAW="$OUTPUTS/tokens_mewsli_finetuning"
WORKDIR="$OUTPUTS/workdirs/v2_retraining_with_model_from_all"
N_OF_ROUNDS=10

run_ml_finetuning_round() {
    local DAMUEL_DESCS_TOKENS_RAW=$1
    local DAMUEL_LINKS_TOKENS_RAW=$2
    local MEWSLI_TOKENS_RAW=$3
    local WORKDIR=$4
    local STATE_DICT=${5:-"None"}
    local ROUND_ID=${6:-"0"}
    local N_OF_ROUNDS=${7}

    local STEPS_PER_EPOCH=1000

    # Multiple by 2 to make sure that if a link contained something faulty we can skip it.
    local LINKS_PER_ROUND=$(($STEPS_PER_EPOCH * 1000 * 3000))
    echo "LPR $LINKS_PER_ROUND"

    local ACTION_SCRIPT="run_action_gin.py $MODEL_CONFIG_PATH $TRAIN_CONFIG_PATH"

    ENV="../venv/bin/activate"
    source $ENV

    # ====================TOKENS COPY====================

    local DAMUEL_LINKS_TOKENS="$WORKDIR/damuel_links_together_tokens_$ROUND_ID"
    if [ ! "$(ls -A $DAMUEL_LINKS_TOKENS)" ]; then
        ../venv/bin/python $ACTION_SCRIPT "copy" \
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
        ../venv/bin/python $ACTION_SCRIPT "embs_from_tokens_model_name_and_state_dict" \
            --source_path="$DAMUEL_DESCS_TOKENS_RAW" \
            --dest_path="$DAMUEL_FOR_INDEX_DIR" \
            --state_dict_path="$STATE_DICT"
    fi

    # ====================DAMUEL LINKS EMBEDDING====================

    local DAMUEL_LINKS_DIR="$WORKDIR/links_embs_$ROUND_ID"

    mkdir -p "$DAMUEL_LINKS_DIR"

    if [ ! "$(ls -A $DAMUEL_LINKS_DIR)" ]; then
        echo "Running embs generating for damuel links"
        ../venv/bin/python $ACTION_SCRIPT "embed_links_for_generation" \
            --links_tokens_dir_path="$DAMUEL_LINKS_TOKENS" \
            --dest_dir_path="$DAMUEL_LINKS_DIR" \
            --state_dict_path="$STATE_DICT"
    fi

    # ====================GENERATING BATCHES====================

    local BATCH_DIR="$WORKDIR/batches_$ROUND_ID"

    mkdir -p "$BATCH_DIR"
    if [ ! "$(ls -A $BATCH_DIR)" ]; then
        echo "Running batches generating for damuel"
        #../venv/bin/python -m cProfile -o "generate.prof" $ACTION_SCRIPT "generate" \
        ../venv/bin/python $ACTION_SCRIPT "generate" \
            --LINKS_EMBS_DIR="$DAMUEL_LINKS_DIR" \
            --INDEX_TOKENS_DIR="$DAMUEL_DESCS_TOKENS_RAW" \
            --INDEX_EMBS_QIDS_DIR="$DAMUEL_FOR_INDEX_DIR" \
            --OUTPUT_DIR="$BATCH_DIR"
    fi

    # ====================TRAINING MODEL====================

    local MODELS_DIR="$WORKDIR/models_$ROUND_ID"

    mkdir -p $MODELS_DIR

    if [ ! "$(ls -A $MODELS_DIR)" ]; then
        echo "Running training for damuel"
        #../venv/bin/python -m cProfile -o "train_ddp.prof" $ACTION_SCRIPT "train_ddp" \
        ../venv/bin/python $ACTION_SCRIPT "train_ddp" \
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
        ../venv/bin/python $ACTION_SCRIPT "embs_from_tokens_model_name_and_state_dict" \
            --source_path="$DAMUEL_DESCS_TOKENS_RAW" \
            --dest_path="$DAMUEL_FOR_INDEX_NEW_DIR" \
            --state_dict_path="$MODELS_DIR/final.pth"
    fi

    local LANGUAGES=("ar" "de" "en" "es" "ja" "fa" "sr" "ta" "tr")

    for LANG in "${LANGUAGES[@]}"; do
        echo "Processing language: $LANG"
        
        local LANG_TOKEN_DIR="$MEWSLI_TOKENS_RAW/$LANG"
        local MEWSLI_EMBS_DIR="$WORKDIR/mewsli_embs_${LANG}_$ROUND_ID"
        
        mkdir -p "$MEWSLI_EMBS_DIR"
        
        if [ ! "$(ls -A $MEWSLI_EMBS_DIR)" ]; then
            echo "Running embs generating for mewsli - Language: $LANG"
            ../venv/bin/python $ACTION_SCRIPT "embs_from_tokens_model_name_and_state_dict" \
                --source_path="$LANG_TOKEN_DIR" \
                --dest_path="$MEWSLI_EMBS_DIR" \
                --state_dict_path="$MODELS_DIR/final.pth"
        fi
        
        ../venv/bin/python $ACTION_SCRIPT "recalls" \
            --damuel_dir="$DAMUEL_FOR_INDEX_NEW_DIR" \
            --mewsli_dir="$MEWSLI_EMBS_DIR"
        
        echo "Completed processing for language: $LANG"
        echo "----------------------------------------"
    done

    rm -r $BATCH_DIR $DAMUEL_LINKS_DIR $DAMUEL_FOR_INDEX_DIR
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

for ((ROUND_ID=3; ROUND_ID<$N_OF_ROUNDS; ROUND_ID++))
do
    run_ml_finetuning_round "$DAMUEL_DESCS_TOKENS_RAW" "$DAMUEL_LINKS_TOKENS_RAW" \
        "$MEWSLI_TOKENS_RAW" \
        "$WORKDIR" "$STATE_DICT" \
        "$ROUND_ID" "$N_OF_ROUNDS"
    #if [ ! -e "$WORKDIR/models_$ROUND_ID/final.pth" ]; then
    #    echo "Running round $ROUND_ID"

    #    run_ml_finetuning_round "$DAMUEL_DESCS_TOKENS_RAW" "$DAMUEL_LINKS_TOKENS_RAW" \
    #        "$MEWSLI_TOKENS_RAW" \
    #        "$WORKDIR" "$STATE_DICT" \
    #        "$ROUND_ID" "$N_OF_ROUNDS"
    #fi

    STATE_DICT="$WORKDIR/models_$ROUND_ID/final.pth"
done
