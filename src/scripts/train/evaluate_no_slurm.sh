set -ueo pipefail

cd ../../

echo "Running all_langs.sh"
echo "Current directory: $(pwd)"

MODEL_CONFIG_PATH="../configs/lealla_m.gin"
TRAIN_CONFIG_PATH="../configs/train.gin"

DAMUEL_FOR_INDEX_NEW_DIR="$OUTPUTS/triplets/combined"
MEWSLI_TOKENS_RAW="$OUTPUTS/tokens_mewsli_finetuning"
WORKDIR="$OUTPUTS/workdirs/all"
ROUND_ID=7
MODELS_DIR="$WORKDIR/models_$ROUND_ID"

ACTION_SCRIPT="run_action_gin.py $MODEL_CONFIG_PATH $TRAIN_CONFIG_PATH"

LANGUAGES=("ar" "de" "en" "es" "ja" "fa" "sr" "ta" "tr")

for LANG in "${LANGUAGES[@]}"; do
    echo "Processing language: $LANG"
    
    LANG_TOKEN_DIR="$MEWSLI_TOKENS_RAW/$LANG"
    MEWSLI_EMBS_DIR="$WORKDIR/mewsli_embs_${LANG}_$ROUND_ID"
    
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