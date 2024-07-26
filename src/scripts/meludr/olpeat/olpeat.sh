#!/bin/bash

set -ueo pipefail

languages=(ar de en es fa ja sr ta tr)

for lang in "${languages[@]}"; do
    ./olpeat_lang.sh "$lang" &
done