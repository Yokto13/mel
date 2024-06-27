#!/bin/bash

set -ueo pipefail
# assumes spacy venv sourced
# installs 
    # "de": "de_core_news_sm",
    # "en": "en_core_web_sm",
    # "es": "es_core_news_sm",
    # "ja": "ja_core_news_sm",

python3 -m spacy download en_core_web_sm
python3 -m spacy download de_core_news_sm
python3 -m spacy download es_core_news_sm
python3 -m spacy download ja_core_news_sm