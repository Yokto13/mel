#!/bin/bash

# Tokenizes mentions for both DaMuEL and Mewsli.
# This tokenization works only on mentions which makes it suitable with OLPEAT.

set -ueo pipefail

cd ../../

source ../venv/bin/activate

python run_action_gin.py ../configs/lealla.gin ../configs/tokenization_mentions.gin run_damuel_mention

python run_action_gin.py ../configs/lealla.gin ../configs/tokenization_mentions.gin run_mewsli_mention
