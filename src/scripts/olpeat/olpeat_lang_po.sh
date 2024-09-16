#!/bin/bash

# Replicated OLPEAT from MELUDR.

set -ueo pipefail

LANG="$1"

DAMUEL_INPUT="$OUTPUTS/embs_damuel_at_po/$LANG"
DAMUEL_TOKENS="$OUTPUTS/tokens_damuel_at/$LANG/all"
DESCS="$DAMUEL_INPUT/descs/embs_tokens.npz"
DESCS_PAGES="$DAMUEL_INPUT/descs_pages/embs_tokens.npz"
LINKS="$DAMUEL_INPUT/links/embs_tokens.npz"
MEWSLI_INPUT="$OUTPUTS/embs_mewsli_at_po/$LANG/embs_qids.npz"

cd ../embs

# sbatch --wait -p "gpu-troja,gpu-ms" -G 8 -C "gpuram16G" --mem=220G damuel_at_lang_v2.sh "$LANG" "pooler_output"
# sbatch --wait -p "gpu-troja,gpu-ms" -G 1 -C "gpuram16G" --mem=30G damuel_at_lang_v2.sh "$LANG"
# sbatch --wait -p "cpu-ms" -c10 --mem=120G damuel_at_lang_v2.sh "$LANG"
sbatch --wait -p "gpu-troja,gpu-ms" -G 1 -C "gpuram16G" mewsli_at_lang.sh "$LANG" "pooler_output"

cd ../olpeat

# ./olpeat_wrap.sh "$DESCS" "$DAMUEL_TOKENS" "$MEWSLI_INPUT" 1 "$LINKS"
run sbatch -c25 --mem=200G ./olpeat_wrap.sh "$DESCS" "$DAMUEL_TOKENS" "$MEWSLI_INPUT" 1 "$LINKS"
run sbatch -c25 --mem=200G ./olpeat_wrap.sh "$DESCS" "$DAMUEL_TOKENS" "$MEWSLI_INPUT" 10 "$LINKS"
