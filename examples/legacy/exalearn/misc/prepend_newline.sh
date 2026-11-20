#!/bin/bash

# Check if the directory is provided as an argument
if [ $# -ne 1 ]; then
    echo "Usage: $0 <directory>"
    exit 1
fi

directory="$1"

# Recursively find and process all files
find "$directory" -type f -exec sed -i 's/Temp for Darshan,/\n&/g' {} +

echo "Adding newlines before patterns in files completed."
