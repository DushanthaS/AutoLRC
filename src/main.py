import os
import sys
import logging
import asyncio
from datetime import datetime
from config_loader import load_config, DEFAULT_INPUT_PATH, DEFAULT_OUTPUT_PATH, DEFAULT_LOGS_PATH, SUPPORTED_LANGUAGES
from processing import batch_process_async, process_audio_file_async

def setup_logging():
    """Set up logging with timestamp-based log file."""
    os.makedirs(DEFAULT_LOGS_PATH, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(DEFAULT_LOGS_PATH, f"transcription_{timestamp}.log")
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    logging.info(f"✅ Log file created: {log_file}")
    return log_file

def print_supported_languages():
    """Print list of supported languages."""
    print("\nSupported Languages:")
    print("-------------------")
    for lang, code in SUPPORTED_LANGUAGES.items():
        print(f"{lang} ({code})")

async def main_async():
    try:
        # Set up logging first
        log_file = setup_logging()
        logging.info("Starting transcription process...")
        
        # Load configuration
        config = load_config()
        if not config:
            sys.exit(1)
        logging.info("Configuration loaded successfully")
        
        # Create output directory if it doesn't exist
        os.makedirs(DEFAULT_OUTPUT_PATH, exist_ok=True)
        logging.info(f"Output directory: {DEFAULT_OUTPUT_PATH}")
        
        # Process files
        if os.path.isdir(DEFAULT_INPUT_PATH):
            logging.info(f"Processing directory: {DEFAULT_INPUT_PATH}")
            await batch_process_async(DEFAULT_INPUT_PATH, config)
        elif os.path.isfile(DEFAULT_INPUT_PATH) and DEFAULT_INPUT_PATH.lower().endswith((".mp3", ".wav", ".m4a", ".ogg")):
            logging.info(f"Processing single file: {DEFAULT_INPUT_PATH}")
            await process_audio_file_async(DEFAULT_INPUT_PATH, config)
        else:
            logging.error(f"❌ Invalid input path: {DEFAULT_INPUT_PATH}")
            sys.exit(1)
            
    except Exception as e:
        logging.error(f"❌ Fatal error: {str(e)}", exc_info=True)
        sys.exit(1)

def main():
    """Main entry point."""
    if len(sys.argv) > 1 and sys.argv[1] == "--list-languages":
        print_supported_languages()
        sys.exit(0)
        
    asyncio.run(main_async())

if __name__ == "__main__":
    main() 