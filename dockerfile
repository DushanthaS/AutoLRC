# Base image supports Nvidia CUDA but does not require it and can also run demucs on the CPU
FROM nvidia/cuda:12.6.2-base-ubuntu22.04

USER root
ENV TORCH_HOME=/data/models
ENV OMP_NUM_THREADS=1

# Install required tools
# Notes:
#  - build-essential and python3-dev are included for platforms that may need to build some Python packages (e.g., arm64)
#  - torchaudio >= 0.12 now requires ffmpeg on Linux
RUN apt update && apt install -y --no-install-recommends \
    build-essential \
    ffmpeg \
    git \
    python3 \
    python3-dev \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Set up a working directory
WORKDIR /app

# Install Demucs (now maintained in the original author's github space)
RUN git clone --single-branch --branch main https://github.com/adefossez/demucs /lib/demucs

# Install dependencies with overrides for known working versions on this base image
WORKDIR /lib/demucs
RUN python3 -m pip install -e . "torch<2" "torchaudio<2" "numpy<2" --no-cache-dir

# Download the default model by running a test
RUN python3 -m demucs -d cpu test.mp3 
# Cleanup output - we just used this to download the model
RUN rm -r separated

# Install additional dependencies needed for the transcription script
RUN python3 -m pip install pydub google-generativeai

# Create necessary directories
RUN mkdir -p /data/models /config

# Copy scripts and config
COPY src/ /app/
COPY config/ /config/

# Set proper permissions
RUN chmod 644 /config/autolrc_config.json

# Create and set up entrypoint script
RUN echo '#!/bin/bash\n\
if [ ! -f "/config/autolrc_config.json" ]; then\n\
    echo "Error: Configuration file not found at /config/autolrc_config.json"\n\
    exit 1\n\
fi\n\
\n\
# Run the transcription script with the provided arguments\n\
python3 /app/transcribe.py "$@"\n\
' > /app/entrypoint.sh

# Make the entrypoint script executable
RUN chmod +x /app/entrypoint.sh

# Define volume mounts
VOLUME /data/models
VOLUME /config

# Set working directory for running commands
WORKDIR /app

# Use the entrypoint script
ENTRYPOINT ["/app/entrypoint.sh"]
