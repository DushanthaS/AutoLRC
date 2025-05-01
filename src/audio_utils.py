import os
import re
import shutil
import logging
from pydub import AudioSegment
from pydub.utils import mediainfo
from config_loader import TEMP_DIR, DEMUCS_OUTPUT_FOLDER, DEFAULT_LOGS_PATH, DEFAULT_INPUT_PATH, DEFAULT_OUTPUT_PATH

def get_audio_duration(path):
    """Get the duration of an audio file in seconds"""
    try:
        return float(mediainfo(path)['duration'])
    except Exception as e:
        logging.error(f"❌ Failed to get audio duration: {e}")
        return 0.0

def convert_to_wav(input_path):
    """Convert audio to 16kHz mono WAV format for API compatibility"""
    logging.info("🔄 Converting to API-compatible WAV format...")
    try:
        audio = AudioSegment.from_file(input_path)
        audio = audio.set_frame_rate(16000).set_channels(1)
        output_path = os.path.join(TEMP_DIR, os.path.splitext(os.path.basename(input_path))[0] + "_converted.wav")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        audio.export(output_path, format="wav", parameters=["-ac", "1", "-ar", "16000"])
        return output_path
    except Exception as e:
        logging.error(f"❌ Conversion failed: {e}")
        return None

def sanitize_filename(filename):
    """Create a safe version of the filename with only ASCII characters"""
    # Remove special characters and replace spaces with underscores
    safe_name = re.sub(r'[^a-zA-Z0-9_-]', '_', filename)
    # Remove multiple consecutive underscores
    safe_name = re.sub(r'_+', '_', safe_name)
    return safe_name

def cleanup_temp_files(converted_paths=None):
    """Clean up temporary files and folders"""
    if converted_paths:
        for path in converted_paths:
            if os.path.exists(path):
                try:
                    os.remove(path)
                    logging.info(f"✅ Removed temporary file: {path}")
                except Exception as e:
                    logging.warning(f"⚠️ Failed to remove file: {e}")
    
    if os.path.exists(TEMP_DIR):
        try:
            shutil.rmtree(TEMP_DIR)
            logging.info(f"✅ Removed temp directory: {TEMP_DIR}")
        except Exception as e:
            logging.warning(f"⚠️ Failed to remove folder: {e}")
    
    if os.path.exists(DEMUCS_OUTPUT_FOLDER):
        try:
            shutil.rmtree(DEMUCS_OUTPUT_FOLDER)
            logging.info(f"✅ Removed demucs output folder: {DEMUCS_OUTPUT_FOLDER}")
        except Exception as e:
            logging.warning(f"⚠️ Failed to remove folder: {e}") 