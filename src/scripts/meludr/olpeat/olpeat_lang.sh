#!/bin/bash

set -ueo pipefail

LANG="$1"

DAMUEL_INPUT="$OUTPUTS/embs_damuel_at/$LANG"
DESCS="$DAMUEL_INPUT/descs"
LINKS="$DAMUEL_INPUT/links"
MEWSLI_INPUT="$OUTPUTS/embs_mewsli_at/$LANG"

cd ../../embs

sbatch --wait -p gpu-troja -G 1 -C "gpuram16G" mewsli_at_lang.sh $LANG
sbatch --wait -p gpu-troja -G 8 -C "gpuram16G" damuel_at_lang.sh $LANG

cd ../../

source ../venv/bin/activate

sbatch -c30 --mem=200G python run_action.py "meludr_olpeat" $DESCS $MEWSLI_INPUT 1 true $LINKS
sbatch -c30 --mem=200G python run_action.py "meludr_olpeat" $DESCS $MEWSLI_INPUT 10 true $LINKS