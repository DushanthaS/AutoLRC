#!/bin/bash
set -e

# Default configuration
IMAGE_NAME="autolrc"
INPUT_DIR="./input"
OUTPUT_DIR="./output"
CONFIG_DIR="./config"
MODELS_DIR="./models"
LOGS_DIR="./logs"
USE_GPU=false
API_KEY=""

# Function to display usage
show_usage() {
  echo "Sinhala Audio Transcription Tool Launcher"
  echo "-----------------------------------------"
  echo "Usage: ./run.sh [OPTIONS]"
  echo ""
  echo "Options:"
  echo "  --input, -i PATH              Input folder or audio file path (default: ./input)"
  echo "  --output, -o PATH             Output folder for transcription files (default: ./output)"
  echo "  --config, -c PATH             Config directory (default: ./config)"
  echo "  --models, -m PATH             Models cache directory (default: ./models)"
  echo "  --api-key, -k KEY             Gemini API key"
  echo "  --gpu                         Enable GPU acceleration"
  echo "  --no-vocal-isolation          Skip vocal isolation"
  echo "  --no-lrc                      Don't generate LRC files"
  echo "  --no-txt                      Don't generate text files"
  echo "  --build                       Build Docker image before running"
  echo "  --help, -h                    Show this help message"
  echo ""
  echo "Example:"
  echo "  ./run.sh --input ~/Music/sinhala --output ~/transcriptions --api-key YOUR_KEY"
  exit 0
}

# Parse command line arguments
VOCAL_ISOLATION="--vocal-isolation"
CREATE_LRC="--lrc"
CREATE_TXT="--txt"
BUILD_IMAGE=false

while [[ "$#" -gt 0 ]]; do
  case $1 in
    --input|-i) INPUT_DIR="$2"; shift ;;
    --output|-o) OUTPUT_DIR="$2"; shift ;;
    --config|-c) CONFIG_DIR="$2"; shift ;;
    --models|-m) MODELS_DIR="$2"; shift ;;
    --api-key|-k) API_KEY="$2"; shift ;;
    --gpu) USE_GPU=true ;;
    --no-vocal-isolation) VOCAL_ISOLATION="--no-vocal-isolation" ;;
    --no-lrc) CREATE_LRC="--no-lrc" ;;
    --no-txt) CREATE_TXT="--no-txt" ;;
    --build) BUILD_IMAGE=true ;;
    --help|-h) show_usage ;;
    *) echo "Unknown parameter: $1"; show_usage ;;
  esac
  shift
done

# Create directories if they don't exist
mkdir -p "$INPUT_DIR"
mkdir -p "$OUTPUT_DIR"
mkdir -p "$CONFIG_DIR"
mkdir -p "$MODELS_DIR"
mkdir -p "$LOGS_DIR"

# Build image if requested
if [ "$BUILD_IMAGE" = true ]; then
  echo "Building Docker image..."
  docker build -t "$IMAGE_NAME" .
fi

# Prepare GPU parameter
GPU_PARAM=""
if [ "$USE_GPU" = true ]; then
  GPU_PARAM="--gpus all"
fi

# Print debug information
echo "Debug Information:"
echo "Input Directory: $INPUT_DIR"
echo "Output Directory: $OUTPUT_DIR"
echo "Config Directory: $CONFIG_DIR"
echo "Models Directory: $MODELS_DIR"
echo "Logs Directory: $LOGS_DIR"
echo "API Key Provided: $([ -n "$API_KEY" ] && echo "Yes" || echo "No")"
echo "GPU Enabled: $([ "$USE_GPU" = true ] && echo "Yes" || echo "No")"
echo "Vocal Isolation: $([ "$VOCAL_ISOLATION" = "--vocal-isolation" ] && echo "Enabled" || echo "Disabled")"
echo "Create LRC: $([ "$CREATE_LRC" = "--lrc" ] && echo "Yes" || echo "No")"
echo "Create TXT: $([ "$CREATE_TXT" = "--txt" ] && echo "Yes" || echo "No")"

# Run the container
echo "Starting transcription..."
docker run --rm $GPU_PARAM \
  -v "$(realpath "$INPUT_DIR"):/app/input" \
  -v "$(realpath "$OUTPUT_DIR"):/app/output" \
  -v "$(realpath "$CONFIG_DIR"):/app/config" \
  -v "$(realpath "$MODELS_DIR"):/app/models" \
  -v "$(realpath "$LOGS_DIR"):/app/logs" \
  "$IMAGE_NAME"

echo "Transcription completed. Results are in $OUTPUT_DIR"
echo "Logs are available in $LOGS_DIR"
