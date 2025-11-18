#!/bin/bash

# Tokenizes mentions for both DaMuEL and Mewsli.
# This tokenization works only on mentions which makes it suitable with OLPEAT.

set -ueo pipefail

cd ../../

uv run python run_action_gin.py ../configs/paraphrase_m.gin ../configs/tokenization_context_paraphrase_old_damuel.gin run_damuel_description_context
uv run python run_action_gin.py ../configs/paraphrase_m.gin ../configs/tokenization_context_paraphrase_old_damuel.gin run_damuel_link_context

