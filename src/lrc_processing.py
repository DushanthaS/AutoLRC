"""
LRC Generation Module

This module provides functions to generate LRC (Lyrics) files from audio files
and their corresponding transcripts.
"""

import os
import logging
from .audio_timestamp_analyzer import create_lrc_file_async

async def generate_lrc_from_transcript_async(audio_path, transcript_path, output_path):
    """
    Generate an LRC file from an audio file and its transcript.
    
    Args:
        audio_path (str): Path to the audio file
        transcript_path (str): Path to the transcript file
        output_path (str): Directory where the LRC file will be saved
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Ensure transcript file exists
        if not os.path.isfile(transcript_path):
            logging.error(f"❌ Transcript file not found: {transcript_path}")
            return False
        
        # Read transcript content
        with open(transcript_path, 'r', encoding='utf-8') as f:
            transcript_content = f.read()
        
        # Get base filename
        base_filename = os.path.basename(audio_path)
        lrc_filename = os.path.splitext(base_filename)[0] + ".lrc"
        
        # Create LRC file
        result = await create_lrc_file_async(audio_path, transcript_content, output_path, lrc_filename)
        
        return result
        
    except Exception as e:
        logging.error(f"❌ Error generating LRC file: {e}", exc_info=True)
        return False