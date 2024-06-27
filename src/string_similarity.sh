#!/bin/bash

set -ueo pipefail

damuel="/home/farhand/damuel_spark_workdir/damuel_1.0_ta"
mewsli="/home/farhand/bc/data/mewsli/mewsli-9/output/dataset/ta/mentions.tsv"
only_wiki_links=true
R=1
workers=32

VENV=../venv/bin/activate

PARAMETERS="$damuel $mewsli $only_wiki_links $R $workers"

source $VENV
python run_action.py string_similarity $PARAMETERS 

echo "Job completed."

