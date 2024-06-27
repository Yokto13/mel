#!/bin/bash

set -ueo pipefail

SCRIPT_PATH=/home/farhand/bc/src/experiments/alias_table/one_language.py
UNIQUENAME="alias_table_R${1}_lang_${2}"

damuel="/home/farhand/damuel_spark_workdir/damuel_1.0_${2}"
mewsli="/home/farhand/bc/data/mewsli/mewsli-9/output/dataset/${2}/mentions.tsv"
only_wiki_links=true
R=$1
xz=true
max_per_qid=10000000000000000000000000000000000
lower=true

ENV=/home/farhand/bc/venv/bin/activate

ACTION_SCRIPT="/home/farhand/bc/src/run_finetuning_action.py"

ID="${UNIQUENAME}_$(date +'%Y%m%d_%H%M')"

# Unique job directory name - using date and time for uniqueness
PARAMETERS="$damuel $mewsli $only_wiki_links $R $xz $max_per_qid $lower"

echo "Running from $PWD"

source $ENV
python $ACTION_SCRIPT "at_lemmas" $PARAMETERS 

echo "Job completed. Job details are in $JOB_DIR."

