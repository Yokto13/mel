#!/bin/bash

set -ueo pipefail

cd ../../

VENV=../venv/bin/activate
source $VENV

RESULT="$OUTPUTS/tokens_damuel_finetuning"

python run_action.py tokens_for_all_damuel_finetuning_pages "$DAMUEL/1.0" "$RESULT" 100 64 "/lnet/work/home-students-external/farhan/troja/outputs/models/LEALLA-small"
