#!/bin/bash

# Replicated OLPEAT from MELUDR.

set -ueo pipefail

LANG="$1"

DAMUEL_INPUT="$OUTPUTS/embs_damuel_at/$LANG"
DESCS="$DAMUEL_INPUT/descs"
DESCS_PAGES="$DAMUEL_INPUT/descs_pages"
LINKS="$DAMUEL_INPUT/links"
MEWSLI_INPUT="$OUTPUTS/embs_mewsli_at/$LANG"

cd ../../embs

sbatch --wait -p gpu-troja,gpu-ms -G 1 -C "gpuram16G" mewsli_at_lang.sh "$LANG"
sbatch --wait -p gpu-troja,gpu-ms -G 8 -C "gpuram16G" --mem=220G damuel_at_lang.sh "$LANG"

cd ../meludr/olpeat

# run sbatch -c30 --mem=200G ./olpeat_wrap.sh $DESCS $MEWSLI_INPUT 1 true "$LINKS"
run sbatch -c25 --mem=100G ./olpeat_wrap.sh "$DESCS_PAGES" "$MEWSLI_INPUT" "$LINKS"
run sbatch -c25 --mem=100G ./olpeat_wrap.sh "$DESCS" "$MEWSLI_INPUT" "$LINKS"
