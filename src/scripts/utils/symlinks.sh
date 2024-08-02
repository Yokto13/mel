#!/bin/bash

set -ueo pipefail

# Check if a path is provided
if [ $# -eq 0 ]; then
    echo "Please provide a path as an argument."
    exit 1
fi

# Store the provided path
BASE_PATH="$1"

# Check if the provided path exists and is a directory
if [ ! -d "$BASE_PATH" ]; then
    echo "The provided path is not a valid directory."
    exit 1
fi

# Function to create symlinks
create_symlinks() {
    local source_dir="$1"
    local target_dir="$2"
    
    if [ -d "$source_dir" ]; then
        for item in "$source_dir"/*; do
            if [ -e "$item" ]; then
                echo "$item"
                local basename=$(basename "$item")
                local target_path="$target_dir/$basename"
                ln -sf "$item" "$target_path"
            fi
        done
    fi
}

# Iterate through all directories in the base path
for d in "$BASE_PATH"/*/; do
    if [ -d "$d" ]; then
        # Create the 'all' directory
        mkdir -p "${d}all"
        
        # Create symlinks from 'descs' to 'all'
        create_symlinks "${d}descs" "${d}all"
        
        # Create symlinks from 'links' to 'all'
        create_symlinks "${d}links" "${d}all"
    fi
done

echo "Process completed successfully."