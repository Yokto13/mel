#!/bin/bash

set -ueo pipefail

SCRIPT_PATH=/home/farhand/bc/src/finetunings/evaluation/entities.py

MEWSLI=$1
DAMUEL_ENTITIES=$2
R=$3
USE_SCANN=true

UNIQUENAME="no_training_scann_R${1}_base"

# data_path="/home/farhand/bc_jobs_important/damuel_links_tokens"

ENV=/home/farhand/bc/src/experiments/sentence_transformers/venv/bin/activate

ID="${UNIQUENAME}_$(date +'%Y%m%d_%H%M')"

# PARAMETERS="$DAMUEL_ALL $MEWSLI $R $USE_SCANN"
PARAMETERS="$DAMUEL_ENTITIES $MEWSLI $R $USE_SCANN"

echo "Running from $PWD"

source $ENV
python $SCRIPT_PATH $PARAMETERS 

