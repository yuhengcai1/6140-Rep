#!/bin/bash

# Set the input directory and output directory
INPUT_DIR="./data"
OUTPUT_DIR="./output"
SCRIPT_PATH="venv/test.py"  # Updated script path
METHOD="Approx"
CUTOFF_TIME=300

# Loop through all .tsp files in the input directory
for file in "$INPUT_DIR"/*.tsp; do
    # Derive the expected output filename
    filename=$(basename -- "$file")
    output_file="$OUTPUT_DIR/${filename%.*}_${METHOD}_${CUTOFF_TIME}_new.sol"  # Updated output filename

    # Check if the output file already exists
    if [ ! -f "$output_file" ]; then
        echo "Processing $file..."
        python "$SCRIPT_PATH" "$file" "$METHOD" "$CUTOFF_TIME"
    else
        echo "Skipping $file (output already exists: $output_file)"
    fi
done
