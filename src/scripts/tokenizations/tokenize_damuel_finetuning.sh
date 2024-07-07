#!/bin/bash

set -ueo pipefail

VENV=../venv/bin/activate
source $VENV
python run_action.py tokens_for_all_damuel_finetuning "/home/farhand/damuel_spark_workdir" "/home/farhand/tokens_damuel" 128 64 "setu4993/LEALLA-base"
