import os
import re
import logging
from audio_utils import get_audio_duration, convert_to_wav, cleanup_temp_files
from vocal_isolator import isolate_vocals
from gemini_api import get_gemini_transcript
from timestamping import analyze_audio_timestamps, create_lrc_content

async def transcribe_and_timestamp(audio_path, api_key, config):
    """Transcribe audio and generate LRC with accurate timestamps."""
    # Get plain text transcript
    transcript = await get_gemini_transcript(audio_path, api_key, config)
    if not transcript:
        return None
    
    # Analyze audio for timestamps
    timestamps = analyze_audio_timestamps(audio_path)
    if timestamps is None or len(timestamps) == 0:
        return None
    
    # Create LRC content
    return create_lrc_content(transcript, timestamps)

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
            # Run vocal isolation synchronously since it's a local process
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
        
        # Generate LRC content
        lrc_content = await transcribe_and_timestamp(converted_path, api_key, config)
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
            try:
                with open(text_path, "w", encoding="utf-8") as f:
                    f.write(text_content)
                logging.info(f"✅ Created text file: {text_path}")
            except Exception as e:
                logging.error(f"❌ Failed to write text file: {str(e)}")
        
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