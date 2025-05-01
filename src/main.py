import os
import sys
import asyncio
import logging
from datetime import datetime
from config_loader import load_config, DEFAULT_INPUT_PATH, DEFAULT_OUTPUT_PATH, DEFAULT_LOGS_PATH
from processing import process_audio_file_async, batch_process_async

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
    # Set up initial console logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )
    
    asyncio.run(main_async())

if __name__ == "__main__":
    main() 