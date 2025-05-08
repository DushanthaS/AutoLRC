import os
import re
import logging
import asyncio  
from typing import Optional, Tuple
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

from audio_utils import (
    get_audio_duration,
    convert_to_wav,
    cleanup_temp_files,
    sanitize_filename
)
from vocal_isolator import isolate_vocals
from gemini_api import get_gemini_transcript
from forced_alignment import get_alignment
from config_loader import (
    get_api_key,
    DEFAULT_OUTPUT_PATH,
    DEFAULT_CONFIG,
    SUPPORTED_LANGUAGES
)

# Configure logging
logger = logging.getLogger(__name__)

async def transcribe_and_timestamp(
    audio_path: str,
    config: dict
) -> Tuple[Optional[str], Optional[str]]:
    """Get transcript from Gemini API and perform forced alignment."""
    api_key = get_api_key()
    if not api_key:
        logger.error("No API key available")
        return None, None

    # Get transcript from Gemini API
    try:
        transcript = await get_gemini_transcript(audio_path, api_key, config)
        if not transcript:
            logger.error("Failed to get transcript from Gemini API")
            return None, None
    except Exception as e:
        logger.error(f"Transcription failed: {e}")
        return None, None

    try:
        # Get language from config
        language = config.get('LANGUAGE', 'English')
        
        # Perform forced alignment
        lrc_content, elrc_content, _ = get_alignment(audio_path, transcript, language=language)
        
        return lrc_content, elrc_content
    except Exception as e:
        logger.error(f"Alignment failed: {e}")
        return None, None

def _save_output_file(
    content: str,
    output_dir: str,
    filename: str,
    extension: str
) -> Optional[str]:
    """Helper function to save output files."""
    if not content:
        return None
        
    try:
        os.makedirs(output_dir, exist_ok=True)
        safe_filename = sanitize_filename(filename) + f".{extension}"
        output_path = os.path.join(output_dir, safe_filename)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)
            
        logger.info(f"Saved {extension.upper()} file: {output_path}")
        return output_path
    except Exception as e:
        logger.error(f"Failed to save {extension.upper()} file: {e}")
        return None

async def process_audio_file_async(
    file_path: str,
    config: dict,
    output_dir: Optional[str] = None
) -> bool:
    """Process a single audio file asynchronously."""
    output_dir = output_dir or DEFAULT_OUTPUT_PATH
    temp_files = []
    
    try:
        logger.info(f"Processing file: {file_path}")
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        base_name = sanitize_filename(base_name)
        
        # Handle vocal isolation if enabled
        isolated_path = None
        if config.get('USE_VOCAL_ISOLATION', False):
            isolated_path = isolate_vocals(file_path)
            if isolated_path:
                file_path = isolated_path
                temp_files.append(isolated_path)

        # Convert audio to WAV format
        converted_path = convert_to_wav(file_path)
        if not converted_path:
            return False
        temp_files.append(converted_path)
        
        try:
            # Get both LRC and eLRC content
            lrc_content, elrc_content = await transcribe_and_timestamp(converted_path, config)
            
            # Save output files
            outputs_created = False
            
            if config.get('CREATE_LRC', True) and lrc_content:
                if _save_output_file(lrc_content, output_dir, base_name, 'lrc'):
                    outputs_created = True
            
            if config.get('CREATE_ELRC', True) and elrc_content:
                if _save_output_file(elrc_content, output_dir, base_name, 'elrc'):
                    outputs_created = True
            
            if config.get('CREATE_TXT', False) and lrc_content:
                txt_content = re.sub(r'\[\d{2}:\d{2}\.\d{2}\]', '', lrc_content)
                txt_content = re.sub(r'<\d{2}:\d{2}\.\d{2}>', '', txt_content)
                txt_content = '\n'.join(line.strip() for line in txt_content.splitlines() if line.strip())
                
                if _save_output_file(txt_content, output_dir, base_name, 'txt'):
                    outputs_created = True
            
            return outputs_created
            
        finally:
            cleanup_temp_files(temp_files)
            
    except Exception as e:
        logger.error(f"Error processing file {file_path}: {e}")
        return False

async def batch_process_async(
    folder_path: str,
    config: dict,
    output_dir: Optional[str] = None,
    max_workers: int = 4
) -> Tuple[int, int]:
    """Process all audio files in a folder asynchronously."""
    logger.info(f"Processing directory: {folder_path}")
    
    try:
        audio_files = [
            f for f in os.listdir(folder_path) 
            if f.lower().endswith((".mp3", ".wav", ".m4a", ".ogg", ".flac"))
        ]
        
        if not audio_files:
            logger.warning("No audio files found in directory")
            return 0, 0
            
        output_dir = output_dir or DEFAULT_OUTPUT_PATH
        success_count = 0
        
        # Process files with limited concurrency
        semaphore = asyncio.Semaphore(max_workers)
        
        async def process_with_semaphore(file_path):
            async with semaphore:
                return await process_audio_file_async(
                    os.path.join(folder_path, file_path),
                    config,
                    output_dir
                )
        
        tasks = [process_with_semaphore(f) for f in audio_files]
        results = await asyncio.gather(*tasks)
        success_count = sum(1 for result in results if result)
        
        logger.info(f"Processed {success_count}/{len(audio_files)} files successfully")
        return success_count, len(audio_files)
        
    except Exception as e:
        logger.error(f"Batch processing failed: {e}")
        return 0, 0

def print_processing_summary(success_count: int, total_files: int):
    """Print a summary of processing results."""
    print("\n" + "=" * 50)
    print(f"Processing Summary:")
    print(f"- Total files processed: {total_files}")
    print(f"- Successful: {success_count}")
    print(f"- Failed: {total_files - success_count}")
    print("=" * 50 + "\n")

async def process_files(
    input_path: str,
    config: dict,
    output_dir: Optional[str] = None,
    max_workers: int = 4
) -> bool:
    """
    Main processing function that handles both files and directories.
    
    Args:
        input_path: Path to file or directory
        config: Configuration dictionary
        output_dir: Custom output directory
        max_workers: Maximum parallel processes
        
    Returns:
        True if processing succeeded, False otherwise
    """
    try:
        if os.path.isfile(input_path):
            success = await process_audio_file_async(input_path, config, output_dir)
            print_processing_summary(1 if success else 0, 1)
            return success
            
        elif os.path.isdir(input_path):
            success_count, total_files = await batch_process_async(
                input_path, config, output_dir, max_workers
            )
            print_processing_summary(success_count, total_files)
            return success_count > 0
            
        logger.error(f"Invalid input path: {input_path}")
        return False
        
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        return False