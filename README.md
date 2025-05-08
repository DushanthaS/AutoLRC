# AutoLRC

AutoLRC is a powerful tool that automatically generates LRC (Lyric) files from audio files using Google's Gemini API for transcription and Wav2Vec2 for timestamp alignment.

## Features

- 🎵 Automatic transcription of audio files using Google's Gemini API
- ⏱️ Precise timestamp generation using Wav2Vec2 forced alignment
- 🌍 Multilingual support for various languages
- 🎤 Optional vocal isolation for better transcription quality
- 📝 Multiple output formats:
  - LRC (standard lyric format)
  - TXT (plain text transcription)
- 🐳 Docker support for easy deployment
- 🔄 Batch processing of multiple audio files
- 📊 Detailed logging and progress tracking

## Supported Languages

The following languages are supported for transcription and timestamping:

- English (en)
- Sinhala (si)
- Tamil (ta)
- Hindi (hi)
- Spanish (es)
- French (fr)
- German (de)
- Italian (it)
- Japanese (ja)
- Korean (ko)
- Chinese (zh)
- Russian (ru)
- Portuguese (pt)
- Dutch (nl)
- Arabic (ar)

## Prerequisites

- Python 3.8 or higher
- Docker (optional, for containerized deployment)
- Google Gemini API key

## Installation

### Local Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/AutoLRC.git
   cd AutoLRC
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Create a `.env` file in the project root:
   ```bash
   # Create .env file
   echo "GEMINI_API_KEY=your_api_key_here" > .env
   ```

### Docker Installation

1. Build the Docker image:
   ```bash
   docker build -t autolrc .
   ```

2. Create required directories:
   ```bash
   mkdir -p config input output logs
   ```

3. Run the container:
   ```bash
   docker run --gpus all \
     -v "$(pwd)/config:/app/config" \
     -v "$(pwd)/input:/app/input" \
     -v "$(pwd)/output:/app/output" \
     -v "$(pwd)/logs:/app/logs" \
     --env-file .env \
     autolrc
   ```

## Configuration changes

 Modify the `autolrc_config.json` file in the config folder with the following options:

```json
{
    "LANGUAGE": "si",
    "USE_VOCAL_ISOLATION": false,
    "CREATE_LRC": true,
    "CREATE_TXT": true,
    "OUTPUT_PATH": "./output"
}
```
Refer to the `config/autolrc_config_example.json` for more examples. 

### Configuration Options

- `LANGUAGE`: Language code for transcription (default: "en")
- `USE_VOCAL_ISOLATION`: Enable/disable vocal isolation (default: false)
- `CREATE_LRC`: Generate LRC files (default: true)
- `CREATE_TXT`: Generate text files (default: false)
- `OUTPUT_PATH`: Directory for output files (default: "./output")

## Usage

### Command Line

```bash
python src/main.py /path/to/audio/file_or_folder
```

### Docker

```bash
docker run --gpus all \
  -v "$(pwd)/config:/app/config" \
  -v "$(pwd)/input:/app/input" \
  -v "$(pwd)/output:/app/output" \
  -v "$(pwd)/logs:/app/logs" \
  --env-file .env \
  autolrc /app/input/your_audio_file.mp3
```

## Output Files

- `.lrc`: Standard lyric file with line-level timestamps
- `.txt`: Plain text transcription without timestamps

## Environment Variables

The following environment variables can be set in the `.env` file:

- `GEMINI_API_KEY`: Your Google Gemini API key (required)
- `CUDA_VISIBLE_DEVICES`: GPU device selection for Wav2Vec2 (optional)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Google Gemini API for transcription
- Facebook's Wav2Vec2 for forced alignment
- All contributors and users of this project