#!/bin/bash

set -ueo pipefail

SCRIPT_PATH=/home/farhand/bc/src/finetunings/train/train.py
UNIQUENAME="finetuning"

# DATASET_DIR="/home/farhand/bc_jobs/generate_together_tokens_20240401_0147"
DATASET_DIR=$1
MODEL_PATH=$2
EPOCHS=$3
LOGIT_MULTIPLIER=$4
LR=$5
TYPE="mentions"
MODEL_SAVE_DIR=$6
STATE_DICT_PATH=$7

# data_path="/home/farhand/bc_jobs_important/damuel_links_tokens"

ENV=/home/farhand/bc/venv/bin/activate

if [ "$STATE_DICT_PATH" == "None" ]; then
  PARAMETERS="$DATASET_DIR $MODEL_PATH $EPOCHS $LOGIT_MULTIPLIER $LR $TYPE $MODEL_SAVE_DIR"
else
  PARAMETERS="$DATASET_DIR $MODEL_PATH $EPOCHS $LOGIT_MULTIPLIER $LR $TYPE $MODEL_SAVE_DIR $STATE_DICT_PATH"
fi

echo "Running from $PWD"

cd /home/farhand/bc/src

source $ENV
python $SCRIPT_PATH $PARAMETERS

