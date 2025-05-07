# Base image supports Nvidia CUDA but does not require it and can also run demucs on the CPU
FROM nvidia/cuda:11.8.0-base-ubuntu22.04

USER root

# Environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV TORCH_HOME=/data/models/.torch 
# Make venv Python the default for subsequent RUN commands
ENV PATH="/app/venv/bin:${PATH}"

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ffmpeg \
    git \
    curl \
    python3.10 \
    python3.10-dev \
    python3.10-venv \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Create symlink for python3 -> python3.10, and pip
RUN ln -sf /usr/bin/python3.10 /usr/bin/python3 && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

# Set up working directory and Python virtual environment
WORKDIR /app
RUN python3.10 -m venv /app/venv # Use explicit python3.10 to create venv

# The PATH env var above should make these use the venv's pip/python
# Upgrade pip and install wheel, setuptools first in the venv
RUN python3 -m pip install --upgrade pip setuptools wheel

# Install specific PyTorch, Torchaudio (for CUDA 11.8)
RUN python3 -m pip install \
    torch==2.1.0 \
    torchaudio==2.1.0 \
    --index-url https://download.pytorch.org/whl/cu118

# Install specific Transformers and uroman (if uroman is a PyPI package)
RUN python3 -m pip install \
    transformers==4.35.0 

# Install Demucs (now maintained in the original author's github space)
RUN git clone --single-branch --branch main https://github.com/adefossez/demucs /lib/demucs

# Install dependencies with overrides for known working versions on this base image
WORKDIR /lib/demucs
RUN python3 -m pip install -e . "torch<2" "torchaudio<2" "numpy<2" --no-cache-dir

# Download the default model by running a test
RUN python3 -m demucs -d cpu test.mp3 
# Cleanup output - we just used this to download the model
RUN rm -r separated

# Install additional Python dependencies for your application (will use venv)
RUN python3 -m pip install \
    pydub \
    google-generativeai \
    uroman \
    huggingface_hub

# Create necessary directories
RUN mkdir -p /data/models/.torch /app/input /app/output /app/logs /app/config

# Pre-download and cache Wav2Vec2 model
RUN python3 -c "from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor; \
    model_name = 'facebook/wav2vec2-base-960h'; \
    processor = Wav2Vec2Processor.from_pretrained(model_name); \
    model = Wav2Vec2ForCTC.from_pretrained(model_name); \
    processor.save_pretrained('/data/models/.torch/wav2vec2-base-960h'); \
    model.save_pretrained('/data/models/.torch/wav2vec2-base-960h')"

# Copy application scripts
COPY src/ /app/

# DO NOT COPY config file if it's always provided by volume mount
# The /app/config directory is created by mkdir and will be populated by the mount.

# Create and set up entrypoint script
RUN echo '#!/bin/bash\n\
set -e\n\
VENV_PYTHON="/app/venv/bin/python3"\n\
# The following check will now verify the file from the volume mount
if [ ! -f "/app/config/autolrc_config.json" ]; then\n\
    echo "Error: Configuration file not found at /app/config/autolrc_config.json (expected from volume mount)" >&2\n\
    exit 1\n\
fi\n\
echo "Configuration file /app/config/autolrc_config.json found."\n\
echo "Running transcription script with arguments: $@"\n\
exec "$VENV_PYTHON" /app/main.py "$@"\n\
' > /app/entrypoint.sh && \
    chmod +x /app/entrypoint.sh

# Define volume mounts
VOLUME /data/models/.torch # Match TORCH_HOME
VOLUME /app/config         # This directory will be populated by the host mount
VOLUME /app/input
VOLUME /app/output
VOLUME /app/logs

# Set working directory for running commands
WORKDIR /app

# Use the entrypoint script
ENTRYPOINT ["/app/entrypoint.sh"]