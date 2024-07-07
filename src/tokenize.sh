#!/bin/bash

set -ueo pipefail

VENV=../venv/bin/activate
source $VENV
python tokenize_damuel.py
