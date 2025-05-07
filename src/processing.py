import os
import re
import logging
import tempfile
from audio_utils import get_audio_duration, convert_to_wav, cleanup_temp_files, sanitize_filename
from vocal_isolator import isolate_vocals
from gemini_api import get_gemini_transcript
from timestamping import analyze_audio_timestamps, create_lrc_content
from config_loader import get_api_key, DEFAULT_OUTPUT_PATH

async def transcribe_and_timestamp(audio_path, config):
    """Get transcript from Gemini API and analyze audio for timestamps."""
    # Get API key from environment
    api_key = get_api_key()
    if not api_key:
        return None

    # Get transcript from Gemini API
    transcript = await get_gemini_transcript(audio_path, api_key, config)
    if not transcript:
        logging.error("Failed to get transcript from Gemini API")
        return None

    # Create a temporary text file for aeneas
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as text_file:
        text_file.write(transcript)
        text_file_path = text_file.name

    try:
        # Analyze audio for timestamps using the text file
        timestamps, _ = analyze_audio_timestamps(audio_path, text_file_path)
        if not timestamps:
            logging.error("Failed to generate timestamps")
            return None

        # Create LRC content
        lrc_content = create_lrc_content(transcript, timestamps)
        return lrc_content
    finally:
        # Clean up the temporary text file
        try:
            os.unlink(text_file_path)
        except Exception as e:
            logging.warning(f"Failed to clean up temporary text file: {e}")

async def process_audio_file_async(file_path, config):
    """Process a single audio file asynchronously."""
    try:
        logging.info(f"Processing file: {file_path}")
        
        # Get base filename without extension
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        base_name = sanitize_filename(base_name)
        
        # Handle vocal isolation if enabled
        if config.get('USE_VOCAL_ISOLATION', False):
            logging.info("Starting vocal isolation...")
            isolated_path = isolate_vocals(file_path)
            if isolated_path:
                file_path = isolated_path
                logging.info("Vocal isolation completed successfully")
            else:
                logging.warning("Vocal isolation failed, proceeding with original audio")
        
        # Convert audio to WAV format
        logging.info("Converting audio to WAV format...")
        converted_path = convert_to_wav(file_path)
        if not converted_path:
            logging.error("Failed to convert audio to WAV format")
            return False
        
        try:
            # Get transcript and timestamps
            lrc_content = await transcribe_and_timestamp(converted_path, config)
            if not lrc_content:
                logging.error("Failed to generate LRC content")
                return False
            
            # Save LRC file if enabled
            if config.get('CREATE_LRC', True):
                lrc_path = os.path.join(DEFAULT_OUTPUT_PATH, f"{base_name}.lrc")
                with open(lrc_path, 'w', encoding='utf-8') as f:
                    f.write(lrc_content)
                logging.info(f"Saved LRC file: {lrc_path}")
            
            # Create text file if enabled
            if config.get('CREATE_TXT', True):
                txt_path = os.path.join(DEFAULT_OUTPUT_PATH, f"{base_name}.txt")
                # Remove all timestamp patterns [00:00.00] from the content
                txt_content = re.sub(r'\[\d{2}:\d{2}\.\d{2}\]', '', lrc_content)
                # Remove any extra whitespace and empty lines
                txt_content = '\n'.join(line.strip() for line in txt_content.splitlines() if line.strip())
                with open(txt_path, 'w', encoding='utf-8') as f:
                    f.write(txt_content)
                logging.info(f"Saved text file: {txt_path}")
            
            return True
            
        finally:
            # Clean up temporary files
            cleanup_temp_files([converted_path])
            if config.get('USE_VOCAL_ISOLATION', False) and isolated_path:
                cleanup_temp_files([isolated_path])
            
    except Exception as e:
        logging.error(f"Error processing file {file_path}: {e}")
        return False

async def batch_process_async(folder_path, config):
    """Process all audio files in a folder asynchronously"""
    logging.info(f"🔍 Looking for audio files in {folder_path}")
    audio_files = [f for f in os.listdir(folder_path) if f.lower().endswith((".mp3", ".wav", ".m4a", ".ogg"))]
    logging.info(f"📊 Found {len(audio_files)} audio files")
    
    successful = 0
    
    for file in audio_files:
        file_path = os.path.join(folder_path, file)
        if await process_audio_file_async(file_path, config):
            successful += 1
    
    # Final cleanup
    cleanup_temp_files()
    
    logging.info(f"\n🎉 Processing complete!")
    logging.info(f"📊 Files processed successfully: {successful}/{len(audio_files)}")