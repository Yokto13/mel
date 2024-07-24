#!/bin/bash

set -ueo pipefail

cd ../../../

VENV=../venv/bin/activate
source $VENV

python run_action.py "meludr_olpeat" $1 $2 $3 $4 $5