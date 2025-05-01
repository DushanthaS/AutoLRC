#!/bin/bash
set -e

# Check if config file exists
if [ ! -f "/config/autolrc_config.json" ]; then
    echo "Error: Configuration file not found at /config/autolrc_config.json"
    exit 1
fi

# Parse arguments to find input and output paths
input_path=""
output_path=""
other_args=()

while [[ $# -gt 0 ]]; do
    case $1 in
        --input|-i)
            input_path="$2"
            shift 2
            ;;
        --output|-o)
            output_path="$2"
            shift 2
            ;;
        *)
            other_args+=("$1")
            shift
            ;;
    esac
done

# Validate input path
if [ -z "$input_path" ]; then
    echo "Error: Input path is required"
    exit 1
fi

# Validate output path
if [ -z "$output_path" ]; then
    echo "Error: Output path is required"
    exit 1
fi

# Verify input path exists
if [ ! -e "$input_path" ]; then
    echo "Error: Input path does not exist: $input_path"
    exit 1
fi

# Create output directory if it doesn't exist
mkdir -p "$output_path"

# Print debug information
echo "Input path: $input_path"
echo "Output path: $output_path"
echo "Other arguments: ${other_args[@]}"

# Run the transcription script with all arguments
python3 /app/transcribe.py --input "$input_path" --output "$output_path" "${other_args[@]}"
