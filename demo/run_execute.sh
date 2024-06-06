#!/bin/bash

# Check if the correct number of arguments are provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <prompt> <output_folder_name>"
    exit 1
fi

# Assign the arguments to variables
PROMPT=$1
OUTPUT_FOLDER_NAME=$2

# Define the base output path
BASE_OUTPUT_PATH="results/demo"

# Construct the full output path
FULL_OUTPUT_PATH="${BASE_OUTPUT_PATH}/${OUTPUT_FOLDER_NAME}"

# Run the Python script with the provided arguments
python execute.py --prompt "$PROMPT" --model_path models/checkpoint.pth.tar --obj_path data/source_meshes/horse.obj --output_path "$FULL_OUTPUT_PATH" --sigma 5.0