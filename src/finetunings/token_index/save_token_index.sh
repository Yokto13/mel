#!/bin/bash

set -ueo pipefail

SCRIPT_PATH=/home/farhand/bc/src/finetunings/token_index/save_token_index.py

TOKS=$1
EMBS=$2
DEST=$3
MAX_PER_QID=100000000000
MENTIONS=True

# data_path="/home/farhand/bc_jobs_important/damuel_links_tokens"

ENV=/home/farhand/bc/venv/bin/activate

PARAMETERS="$EMBS $TOKS $DEST $MAX_PER_QID $MENTIONS"

# Create a unique job directory
echo "Running from $PWD"

source $ENV
python $SCRIPT_PATH $PARAMETERS

cd /home/farhand/bc/src

echo $PARAMETERS > $DEST/parameters.txt

