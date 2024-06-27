#!/bin/bash

set -ueo pipefail

damuel=""
mewsli=""
only_wiki_links=true
R=1
lower=true

VENV=../venv/bin/activate

PARAMETERS="$damuel $mewsli $only_wiki_links $R $lower"

source $VENV
python run_action.py at_all $PARAMETERS 

echo "Job completed."

