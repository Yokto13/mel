#!/bin/bash

set -ueo pipefail

SCRIPT_PATH=/home/farhand/bc/src/finetunings/embs_generating/build_together_embs.py
UNIQUENAME="damuel_descs_together_embs"

data_path=$1
dest=$2
model_name=$3
state_dict=$4

ENV=/home/farhand/bc/venv/bin/activate

cd /home/farhand/bc/src

# check that state_dict is None
if [ "$state_dict" == "None" ]; then
  PARAMETERS="$data_path $model_name $dest 1 0"
else
  PARAMETERS="$data_path $model_name $dest 1 0 $state_dict"
fi

echo "Running from $PWD"

source $ENV
python $SCRIPT_PATH $PARAMETERS 
