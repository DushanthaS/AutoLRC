# AutoLRC


This Docker container provides a complete environment for transcribing audio files. It combines:

1. **Demucs** - For vocal isolation (optional but recommended for better transcription results)
2. **Google Gemini 2.0** - For accurate Multi language speech-to-text transcription
3. **Output formats** - Generates both plain text and timestamped LRC files

## Features

- Processes audio files in various formats (MP3, WAV, M4A, OGG)
- Optional vocal isolation to improve transcription quality
- Automatic conversion to API-compatible format
- Support for batch processing of multiple files
- Persistent configuration for API keys
- GPU acceleration support via CUDA (when available)
- Command-line interface with flexible options

## Prerequisites

- Docker installed on your system
- A Google Gemini API key

## Building the Docker Image

```bash
git clone https://your-repository-url/AutoLRC.git
cd AutoLRC
docker build -t AutoLRC .
```

## Usage

### Basic Usage

```bash
docker run -v /path/to/input:/data/input \
           -v /path/to/output:/data/output \
           -v /path/to/config:/config \
           AutoLRC transcribe \
           --input /data/input \
           --output /data/output \
           --api-key YOUR_GEMINI_API_KEY
```

### With GPU Support

If you have a compatible NVIDIA GPU with the NVIDIA Container Runtime:

```bash
docker run --gpus all \
           -v /path/to/input:/data/input \
           -v /path/to/output:/data/output \
           -v /path/to/config:/config \
           AutoLRC transcribe \
           --input /data/input \
           --output /data/output \
           --api-key YOUR_GEMINI_API_KEY
```

### All Options

```bash
docker run AutoLRC help
```

Output:
```
Sinhala Audio Transcription Tool
--------------------------------
Usage: docker run [options] AutoLRC [COMMAND]

Commands:
  help                          Show this help message
  transcribe [OPTIONS]          Run transcription with options

Transcription Options:
  --input, -i PATH              Input folder or audio file path
  --output, -o PATH             Output folder for transcription files
  --txt                         Generate text files (default)
  --no-txt                      Don't generate text files
  --lrc                         Generate LRC files (default)
  --no-lrc                      Don't generate LRC files
  --vocal-isolation, -v         Use Demucs for vocal isolation (default)
  --no-vocal-isolation          Skip vocal isolation
  --api-key KEY                 Gemini API key (if not provided, will use key from config)
```

## Volume Mounts

- `/data/input` - Mount your input audio files here
- `/data/output` - Where transcription results will be saved
- `/data/models` - Persistent storage for downloaded models (optional)
- `/config` - Stores your API key and other settings

## Examples

### Process a Single Audio File

```bash
docker run -v /path/to/file.mp3:/data/input/file.mp3 \
           -v /path/to/output:/data/output \
           -v /path/to/config:/config \
           AutoLRC transcribe \
           --input /data/input/file.mp3 \
           --output /data/output \
           --api-key YOUR_GEMINI_API_KEY
```

### Process All Audio Files in a Directory (Without Vocal Isolation)

```bash
docker run -v /path/to/audio/folder:/data/input \
           -v /path/to/output:/data/output \
           -v /path/to/config:/config \
           AutoLRC transcribe \
           --input /data/input \
           --output /data/output \
           --no-vocal-isolation
```

### Generate Only Text Files (No LRC)

```bash
docker run -v /path/to/audio/folder:/data/input \
           -v /path/to/output:/data/output \
           -v /path/to/config:/config \
           AutoLRC transcribe \
           --input /data/input \
           --output /data/output \
           --no-lrc
```

## API Key Storage

After providing your API key once, it will be stored in the `/config` directory. As long as you mount the same directory on subsequent runs, you won't need to provide the key again.

## Notes

- Vocal isolation improves transcription quality but adds processing time
- The first run will download the required Demucs model (~1GB) which will be stored in the `/data/models` volume
- Processing speed depends on your hardware; GPU acceleration will significantly improve Demucs performance
- When using a mounted config directory, your API key is stored securely with restricted permissions

## Troubleshooting

### No Audio Files Found

Make sure your audio files have the correct extensions (.mp3, .wav, .m4a, or .ogg) and are properly mounted in the container.

```bash
# List files in the input directory inside the container
docker run -v /path/to/input:/data/input --rm AutoLRC ls -la /data/input
```

### Demucs Issues

If Demucs is failing to isolate vocals:

1. Try processing without vocal isolation (`--no-vocal-isolation`)
2. Ensure your audio file is not corrupted
3. Check that the container has enough resources (CPU/memory)

### API Key Issues

If you're experiencing authentication errors:

```bash
# Remove existing API key from config
docker run -v /path/to/config:/config --rm --entrypoint bash AutoLRC -c "rm -f /config/config.json"

# Then run again with a new API key
docker run -v /path/to/input:/data/input -v /path/to/output:/data/output -v /path/to/config:/config AutoLRC transcribe --input /data/input --output /data/output --api-key YOUR_NEW_KEY
```

## Advanced Usage

### Using Pre-downloaded Models

To avoid downloading models on each container recreation, mount a persistent directory to `/data/models`:

```bash
docker run -v /path/to/models:/data/models ... AutoLRC transcribe ...
```

### Processing Different Languages

While this tool is optimized for Sinhala, you can modify the prompt in the code to work with other languages:

```bash
# Mount the script into the container and edit it
docker run -v /path/to/your/custom/transcribe.py:/app/transcribe.py ... AutoLRC transcribe ...
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Facebook Research for the Demucs library
- Google for the Gemini API
- All contributors to the open-source libraries used in this project