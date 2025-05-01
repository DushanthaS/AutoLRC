import os
import json
import shutil
import time
import subprocess
import sys
import re
import argparse
import logging
import asyncio
from datetime import datetime
from pathlib import Path
from pydub import AudioSegment
from pydub.utils import mediainfo
import google.generativeai as genai

# Set up initial console logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)


# ========================
# Configuration
# ========================
CONFIG_PATH = "/app/config/autolrc_config.json"
DEFAULT_INPUT_PATH = "/app/input"
DEFAULT_OUTPUT_PATH = "/app/output"
DEFAULT_LOGS_PATH = "/app/logs"

# Default configuration values
DEFAULT_CONFIG = {
    "GEMINI_API_KEY": "",
    "GEMINI_MODEL": "gemini-2.0-flash",
    "GAP_THRESHOLD": 0.2,  # Gap threshold for LRC lines (seconds)
    "MAX_WORDS_PER_LINE": 7,
    "TEMPERATURE": 0.2,
    "TOP_P": 0.8,
    "TOP_K": 40,
    "CANDIDATE_COUNT": 1,
    "MAX_RETRIES": 3,
    "RETRY_DELAY": 5,
    "USE_VOCAL_ISOLATION": True,
    "CREATE_LRC": True,
    "CREATE_TXT": True
}

# Define consistent temp folder names
DEMUCS_OUTPUT_FOLDER = "/tmp/demucs_output"
TEMP_DIR = "/tmp/temp_audio"

# ========================
# Utility Functions
# ========================
def load_config():
    """Load configuration from JSON file with defaults"""
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, 'r') as f:
                user_config = json.load(f)
                # Merge user config with defaults, keeping user values
                config = DEFAULT_CONFIG.copy()
                config.update(user_config)
                return config
        except Exception as e:
            print(f"⚠️ Error loading config: {e}")
            print("⚠️ Using default configuration")
            return DEFAULT_CONFIG
    
    # If config file doesn't exist, create it with defaults
    try:
        os.makedirs(os.path.dirname(CONFIG_PATH), exist_ok=True)
        with open(CONFIG_PATH, 'w') as f:
            json.dump(DEFAULT_CONFIG, f, indent=4)
        os.chmod(CONFIG_PATH, 0o600)  # Secure permissions for API key
        print("✅ Created default configuration file")
        return DEFAULT_CONFIG
    except Exception as e:
        print(f"⚠️ Error creating config file: {e}")
        return DEFAULT_CONFIG


def format_lrc_time(seconds):
    """Format time in LRC format: [mm:ss.xx]"""
    minutes = int(seconds // 60)
    secs = seconds % 60
    return f"[{minutes:02}:{secs:05.2f}]"

def get_audio_duration(path):
    """Get the duration of an audio file in seconds"""
    try:
        return float(mediainfo(path)['duration'])
    except Exception as e:
        print(f"❌ Failed to get audio duration: {e}")
        return 0.0



def sanitize_filename(filename):
    """Create a safe version of the filename with only ASCII characters"""
    # Remove special characters and replace spaces with underscores
    safe_name = re.sub(r'[^a-zA-Z0-9_-]', '_', filename)
    # Remove multiple consecutive underscores
    safe_name = re.sub(r'_+', '_', safe_name)
    return safe_name

# ========================
# Audio Processing
# ========================
def isolate_vocals(input_path):
    """Run Demucs to isolate vocals"""
    print("🎵 Isolating vocals with Demucs...")
    
    # Create a temporary working copy with a safe filename
    original_filename = os.path.basename(input_path)
    file_ext = os.path.splitext(original_filename)[1]
    safe_filename = sanitize_filename(os.path.splitext(original_filename)[0]) + file_ext
    
    os.makedirs(TEMP_DIR, exist_ok=True)
    
    safe_input_path = os.path.join(TEMP_DIR, safe_filename)
    print(f"📝 Creating a working copy with simplified filename: {safe_filename}")
    shutil.copy2(input_path, safe_input_path)
    
    try:
        # Create output folder if it doesn't exist
        os.makedirs(DEMUCS_OUTPUT_FOLDER, exist_ok=True)
        
        # Create a custom output directory for this specific file to avoid conflict
        output_id = f"output_{int(time.time())}"
        custom_output_dir = os.path.join(DEMUCS_OUTPUT_FOLDER, output_id)
        os.makedirs(custom_output_dir, exist_ok=True)

        print(f"🔄 Running Demucs vocal separation...")
        
        # Convert to absolute paths with forward slashes to avoid path issues
        abs_input_path = os.path.abspath(safe_input_path).replace("\\", "/")
        abs_output_dir = os.path.abspath(custom_output_dir).replace("\\", "/")
        
        # Run Demucs - explicitly specify model and stem
        cmd = [
            "python3", "-m", "demucs.separate",
            "--two-stems=vocals", 
            "-n", "htdemucs",
            "--out", abs_output_dir,
            abs_input_path
        ]
        
        print(f"📋 Command: {' '.join(cmd)}")
        
        # Run the command without shell=True for more reliable execution
        result = subprocess.run(
            cmd,
            text=True, 
            capture_output=True,
            shell=False
        )

        print(f"Demucs stdout: {result.stdout}")
        print(f"Demucs stderr: {result.stderr}")

        if result.returncode != 0:
            print(f"❌ Demucs process failed with code {result.returncode}")
            print("⚠️ Vocal isolation failed. Proceeding with original audio...")
            return None

        # Since we explicitly set the model to htdemucs, look in that folder
        base_name = os.path.splitext(os.path.basename(safe_input_path))[0]
        model_dir = "htdemucs"  # We explicitly specified this model
        
        # Try to find the vocals file with proper path handling using Path
        model_output_dir = Path(custom_output_dir) / model_dir / base_name
        if model_output_dir.exists():
            print(f"🔍 Looking for vocals file in: {model_output_dir}")
            
            # Check for a vocals file with various possible names
            potential_stems = ["vocals", "vocals.wav", "vocal", "voice"]
            for stem in potential_stems:
                stem_path = model_output_dir / stem
                if stem_path.exists():
                    print(f"✅ Found vocals file: {stem_path}")
                    return str(stem_path)
                
                # Also check with .wav extension if not already included
                if not stem.endswith(".wav"):
                    stem_path = model_output_dir / f"{stem}.wav"
                    if stem_path.exists():
                        print(f"✅ Found vocals file: {stem_path}")
                        return str(stem_path)
            
            # If we still haven't found it, list all files in the directory
            print("📋 Files found in output directory:")
            for file in model_output_dir.iterdir():
                print(f"  - {file.name}")
                # If the file has vocal in the name, use it
                if "vocal" in file.name.lower() and file.name.endswith(".wav"):
                    print(f"✅ Found vocals file: {file}")
                    return str(file)

        print("❌ Vocals file not found after processing")
        print("⚠️ Proceeding with original audio...")
        return None

    except Exception as e:
        print(f"❌ Vocal isolation failed: {e}")
        print("⚠️ Proceeding with original audio...")
        return None
    finally:
        # Don't delete the temp file yet as we might need it for transcription
        pass

def convert_to_wav(input_path):
    """Convert audio to 16kHz mono WAV format for API compatibility"""
    print("🔄 Converting to API-compatible WAV format...")
    try:
        audio = AudioSegment.from_file(input_path)
        audio = audio.set_frame_rate(16000).set_channels(1)
        output_path = os.path.join(TEMP_DIR, os.path.splitext(os.path.basename(input_path))[0] + "_converted.wav")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        audio.export(output_path, format="wav", parameters=["-ac", "1", "-ar", "16000"])
        return output_path
    except Exception as e:
        print(f"❌ Conversion failed: {e}")
        return None

# ========================
# Transcription Functions
# ========================
def transcribe_with_gemini(audio_path, api_key, config):
    """Transcribe audio using Gemini API"""
    print("🎤 Transcribing with Gemini...")
    max_retries = config.get("MAX_RETRIES", 3)
    retry_delay = config.get("RETRY_DELAY", 5)
    
    # Configure Gemini client
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(config.get("GEMINI_MODEL", "gemini-2.0-flash"))
    
    for attempt in range(max_retries):
        try:
            # Read audio file and prepare content for transcription
            with open(audio_path, "rb") as audio_file:
                audio_data = audio_file.read()
            
            print(f"📡 Sending request to Gemini API (attempt {attempt+1}/{max_retries})...")
            
            # Create content parts
            prompt = "Transcribe this Sinhala audio with punctuation. Return only the transcription text, nothing else."
            
            # Make API call with configurable parameters
            response = model.generate_content(
                contents=[
                    {"text": prompt},
                    {"inline_data": {"mime_type": "audio/wav", "data": audio_data}}
                ],
                generation_config={
                    "temperature": config.get("TEMPERATURE", 0.2),
                    "top_p": config.get("TOP_P", 0.8),
                    "top_k": config.get("TOP_K", 40),
                    "candidate_count": config.get("CANDIDATE_COUNT", 1),
                }
            )
            
            # Extract text from response
            text = response.text
            
            if not text:
                print("⚠️ No transcription returned")
                if attempt < max_retries - 1:
                    print(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    retry_delay *= 2
                    continue
                return None
            
            print(f"✅ Transcription successful ({len(text)} characters)")
            return text
            
        except Exception as e:
            print(f"⚠️ Attempt {attempt+1} failed: {str(e)}")
            if attempt < max_retries - 1:
                print(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
                retry_delay *= 2
            else:
                print("❌ All retry attempts failed")
                return None
    
    return None

def create_timestamped_words(transcription, audio_duration):
    """Create approximate word timings based on transcription and audio duration"""
    if not transcription:
        return []
        
    words = transcription.split()
    avg_word_duration = audio_duration / len(words)
    
    return [{
        "word": word,
        "start": i * avg_word_duration,
        "duration": avg_word_duration
    } for i, word in enumerate(words)]

def create_text_file(transcription, output_path):
    """Generate plain text file from transcription"""
    try:
        logging.info(f"📄 Creating text file at: {output_path}")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(transcription)
        
        if os.path.exists(output_path):
            logging.info(f"✅ Text file created successfully: {output_path}")
            logging.info(f"📊 File size: {os.path.getsize(output_path)} bytes")
        else:
            logging.error(f"❌ Text file was not created: {output_path}")
    except Exception as e:
        logging.error(f"❌ Failed to create text file: {str(e)}", exc_info=True)

def create_lrc_file(timestamped_words, output_path, config):
    """Generate LRC file from timestamped words"""
    try:
        logging.info(f"📝 Creating LRC file at: {output_path}")
        
        # Create output directory
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        logging.info(f"✅ Created output directory: {os.path.dirname(output_path)}")
        
        lines = []
        current_line = []
        current_start = 0.0
        max_words_per_line = config.get("MAX_WORDS_PER_LINE", 7)

        for i, word_data in enumerate(timestamped_words):
            word = word_data["word"]
            start = word_data["start"]
            
            if not current_line:
                current_start = start
                current_line.append(word)
            else:
                if len(current_line) >= max_words_per_line:
                    lines.append((current_start, " ".join(current_line)))
                    current_start = start
                    current_line = [word]
                else:
                    current_line.append(word)

        # Add the last line if there's anything left
        if current_line:
            lines.append((current_start, " ".join(current_line)))

        # Write to file
        with open(output_path, "w", encoding="utf-8") as f:
            for start_time, line_text in lines:
                f.write(f"{format_lrc_time(start_time)}{line_text}\n")
        
        if os.path.exists(output_path):
            logging.info(f"✅ LRC file created successfully: {output_path}")
            logging.info(f"📊 File size: {os.path.getsize(output_path)} bytes")
            logging.info(f"📊 Number of lines: {len(lines)}")
        else:
            logging.error(f"❌ LRC file was not created: {output_path}")
            
    except Exception as e:
        logging.error(f"❌ Failed to create LRC file: {str(e)}", exc_info=True)

async def transcribe_with_gemini_lrc(audio_path, api_key, config, language="Sinhala"):
    """Transcribe audio using Gemini API and generate LRC content directly"""
    print("🎤 Transcribing with Gemini for LRC generation...")
    max_retries = config.get("MAX_RETRIES", 3)
    retry_delay = config.get("RETRY_DELAY", 5)
    
    # Configure Gemini client
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(config.get("GEMINI_MODEL", "gemini-2.0-flash-thinking"))
    
    instructions = f"""You are a highly skilled and meticulous transcription and translation specialist, fluent in {language}. Your sole task is to transcribe spoken {language} audio into an accurate LRC (Lyrics) file format, optimized for karaoke-style display.

Here's how you will operate:
Input: You will receive an audio file containing {language} speech (singing).
Transcription & Translation: Carefully listen to the audio and create a complete and accurate transcription of the lyrics in {language}.
LRC Formatting: Structure the transcription into an LRC file. This requires:
- Each line of {language} text must be preceded by a timestamp in the format [mm:ss.xx], where mm is minutes, ss is seconds, and xx is hundredths of a second.
- The timestamp must indicate the precise moment the lyrics begin to be sung in the audio.
- Timestamps must be sequential and in ascending order.
- When a single sentence is broken into multiple lines for display purposes, ensure each line has its own accurate timestamp.
- The LRC file MUST cover the entire duration of the audio; there should be no missing sections.
- Accuracy is paramount. Listen carefully, and double-check your work to ensure the {language} transcription is error-free.
- Output: You will ONLY provide the complete LRC file content. Do not include any introductory text, explanations, or markdown formatting. The output MUST be plain text in LRC format.
- Consider splitting lines to keep the phrases short"""

    for attempt in range(max_retries):
        try:
            # Read audio file
            with open(audio_path, "rb") as audio_file:
                audio_data = audio_file.read()
            
            print(f"📡 Sending request to Gemini API (attempt {attempt+1}/{max_retries})...")
            
            # Generate LRC content
            response = await model.generate_content_async(
                contents=[
                    {"text": instructions},
                    {"inline_data": {"mime_type": "audio/wav", "data": audio_data}}
                ],
                generation_config={
                    "temperature": config.get("TEMPERATURE", 0.2),
                    "top_p": config.get("TOP_P", 0.8),
                    "top_k": config.get("TOP_K", 40),
                    "candidate_count": config.get("CANDIDATE_COUNT", 1),
                }
            )
            
            # Extract text from response
            lrc_content = response.text.strip()
            
            if not lrc_content:
                print("⚠️ No LRC content returned")
                if attempt < max_retries - 1:
                    print(f"Retrying in {retry_delay} seconds...")
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 2
                    continue
                return None
            
            # Validate LRC format
            if not all(re.match(r'^\[\d{2}:\d{2}\.\d{2}\]', line.strip()) 
                      for line in lrc_content.split('\n') if line.strip()):
                print("⚠️ Invalid LRC format in response")
                if attempt < max_retries - 1:
                    print(f"Retrying in {retry_delay} seconds...")
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 2
                    continue
                return None
            
            print(f"✅ LRC content generated successfully ({len(lrc_content)} characters)")
            return lrc_content
            
        except Exception as e:
            print(f"⚠️ Attempt {attempt+1} failed: {str(e)}")
            if attempt < max_retries - 1:
                print(f"Retrying in {retry_delay} seconds...")
                await asyncio.sleep(retry_delay)
                retry_delay *= 2
            else:
                print("❌ All retry attempts failed")
                return None
    
    return None

# ========================
# Main Processing Function
# ========================
def cleanup_temp_files(converted_paths=None):
    """Clean up temporary files and folders"""
    # Clean up any converted temporary files
    if converted_paths:
        for path in converted_paths:
            if os.path.exists(path):
                try:
                    os.remove(path)
                    print(f"✅ Removed temporary file: {path}")
                except Exception as e:
                    print(f"⚠️ Failed to remove file: {e}")
    
    # Clean up temp directory
    if os.path.exists(TEMP_DIR):
        try:
            shutil.rmtree(TEMP_DIR)
            print(f"✅ Removed temp directory: {TEMP_DIR}")
        except Exception as e:
            print(f"⚠️ Failed to remove folder: {e}")
    
    # Clean up the demucs output folder (only after processing completes)
    if os.path.exists(DEMUCS_OUTPUT_FOLDER):
        try:
            shutil.rmtree(DEMUCS_OUTPUT_FOLDER)
            print(f"✅ Removed demucs output folder: {DEMUCS_OUTPUT_FOLDER}")
        except Exception as e:
            print(f"⚠️ Failed to remove folder: {e}")

async def process_audio_file_async(file_path, api_key, output_dir, config):
    """Process a single audio file asynchronously"""
    try:
        logging.info(f"\n🔊 Processing: {os.path.basename(file_path)}")
        converted_path = None
        vocals_path = None
        temp_files = []
        
        # Get file info
        duration = get_audio_duration(file_path)
        logging.info(f"⏱️ Audio duration: {duration:.2f} seconds")
        
        # Extract and prepare vocals if requested
        if config.get("USE_VOCAL_ISOLATION", True):
            vocals_path = isolate_vocals(file_path)
            if vocals_path:
                audio_path = vocals_path
                temp_files.append(vocals_path)
                logging.info("✅ Using isolated vocals for transcription")
            else:
                audio_path = file_path
                logging.info("⚠️ Using original audio (vocal isolation failed)")
        else:
            audio_path = file_path
            logging.info("ℹ️ Skipping vocal isolation as per configuration")
            
        # Convert to proper format for API
        converted_path = convert_to_wav(audio_path)
        if not converted_path:
            logging.error("❌ Audio conversion failed")
            cleanup_temp_files(temp_files)
            return False
        
        temp_files.append(converted_path)
        logging.info("✅ Audio converted to WAV format")
        
        # Generate base name for output files
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        
        # Generate LRC content directly
        lrc_content = await transcribe_with_gemini_lrc(converted_path, api_key, config)
        if not lrc_content:
            logging.error("❌ LRC generation failed")
            cleanup_temp_files(temp_files)
            return False
            
        # Create LRC file
        lrc_path = os.path.join(output_dir, f"{base_name}.lrc")
        try:
            with open(lrc_path, "w", encoding="utf-8") as f:
                f.write(lrc_content)
            logging.info(f"✅ Created LRC file: {lrc_path}")
        except Exception as e:
            logging.error(f"❌ Failed to write LRC file: {str(e)}")
            cleanup_temp_files(temp_files)
            return False
        
        # Create text file only if explicitly enabled in config
        if config.get("CREATE_TXT", False):
            # Extract text from LRC content by removing timestamps
            text_content = re.sub(r'\[\d{2}:\d{2}\.\d{2}\]', '', lrc_content).strip()
            text_path = os.path.join(output_dir, f"{base_name}.txt")
            create_text_file(text_content, text_path)
            logging.info(f"✅ Created text file: {text_path}")
        
        # Cleanup all temporary files
        cleanup_temp_files(temp_files)
        
        return True
        
    except Exception as e:
        logging.error(f"❌ Processing failed: {str(e)}", exc_info=True)
        return False

async def batch_process_async(folder_path, api_key, output_dir, config):
    """Process all audio files in a folder asynchronously"""
    logging.info(f"🔍 Looking for audio files in {folder_path}")
    audio_files = [f for f in os.listdir(folder_path) if f.lower().endswith((".mp3", ".wav", ".m4a", ".ogg"))]
    logging.info(f"📊 Found {len(audio_files)} audio files")
    
    successful = 0
    
    for file in audio_files:
        file_path = os.path.join(folder_path, file)
        if await process_audio_file_async(file_path, api_key, output_dir, config):
            successful += 1
    
    # Final cleanup
    cleanup_temp_files()
    
    logging.info(f"\n🎉 Processing complete!")
    logging.info(f"📊 Files processed successfully: {successful}/{len(audio_files)}")

# ========================
# Main Execution
# ========================
def setup_logging():
    """Set up logging configuration"""
    try:
        # Create logs directory
        os.makedirs(DEFAULT_LOGS_PATH, exist_ok=True)
        logging.info(f"✅ Created logs directory: {DEFAULT_LOGS_PATH}")
        
        # Create log file with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(DEFAULT_LOGS_PATH, f"transcription_{timestamp}.log")
        
        # Add file handler to existing logger
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logging.getLogger().addHandler(file_handler)
        
        logging.info(f"✅ Log file created: {log_file}")
        return log_file
    except Exception as e:
        logging.error(f"❌ Failed to set up logging: {str(e)}", exc_info=True)
        return None

async def main_async():
    try:
        # Set up logging first
        log_file = setup_logging()
        logging.info("Starting transcription process...")
        
        # Load configuration
        config = load_config()
        logging.info("Configuration loaded successfully")
        
        # Get API key from config
        api_key = config.get("GEMINI_API_KEY", "")
        if not api_key:
            logging.error("❌ No API key provided in config file")
            sys.exit(1)
        
        # Create output directory if it doesn't exist
        os.makedirs(DEFAULT_OUTPUT_PATH, exist_ok=True)
        logging.info(f"Output directory: {DEFAULT_OUTPUT_PATH}")
        
        # Process files
        if os.path.isdir(DEFAULT_INPUT_PATH):
            logging.info(f"Processing directory: {DEFAULT_INPUT_PATH}")
            await batch_process_async(DEFAULT_INPUT_PATH, api_key, DEFAULT_OUTPUT_PATH, config)
        elif os.path.isfile(DEFAULT_INPUT_PATH) and DEFAULT_INPUT_PATH.lower().endswith((".mp3", ".wav", ".m4a", ".ogg")):
            logging.info(f"Processing single file: {DEFAULT_INPUT_PATH}")
            await process_audio_file_async(DEFAULT_INPUT_PATH, api_key, DEFAULT_OUTPUT_PATH, config)
        else:
            logging.error(f"❌ Invalid input path: {DEFAULT_INPUT_PATH}")
            sys.exit(1)
            
    except Exception as e:
        logging.error(f"❌ Fatal error: {str(e)}", exc_info=True)
        sys.exit(1)

def main():
    asyncio.run(main_async())

if __name__ == "__main__":
    main()
