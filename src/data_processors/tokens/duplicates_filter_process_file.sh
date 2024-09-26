#!/bin/bash

set -ueo pipefail 

source=$1
dest_dir=$2

/home/farhand/bc/venv/bin/python /home/farhand/bc/src/data_processors/tokens/duplicates_filter_process_file.py $source $dest_dir