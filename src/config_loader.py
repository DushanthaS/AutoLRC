import os
import json
import logging

# Paths
CONFIG_PATH = "/app/config/autolrc_config.json"
DEFAULT_INPUT_PATH = "/app/input"
DEFAULT_OUTPUT_PATH = "/app/output"
DEFAULT_LOGS_PATH = "/app/logs"

# Temp directories
DEMUCS_OUTPUT_FOLDER = "/tmp/demucs_output"
TEMP_DIR = "/tmp/temp_audio"

# Default configuration values
DEFAULT_CONFIG = {
    "GEMINI_API_KEY": "",
    "GEMINI_MODEL": "gemini-2.0-flash-thinking",
    "LANGUAGE": "English",  # Default language for transcription
    "TEMPERATURE": 0.2,
    "TOP_P": 0.8,
    "TOP_K": 40,
    "CANDIDATE_COUNT": 1,
    "MAX_RETRIES": 3,
    "RETRY_DELAY": 5,
    "USE_VOCAL_ISOLATION": True,
    "CREATE_TXT": False  # Default to not creating text files
}

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
            logging.error(f"⚠️ Error loading config: {e}")
            logging.warning("⚠️ Using default configuration")
            return DEFAULT_CONFIG
    
    # If config file doesn't exist, create it with defaults
    try:
        os.makedirs(os.path.dirname(CONFIG_PATH), exist_ok=True)
        with open(CONFIG_PATH, 'w') as f:
            json.dump(DEFAULT_CONFIG, f, indent=4)
        os.chmod(CONFIG_PATH, 0o600)  # Secure permissions for API key
        logging.info("✅ Created default configuration file")
        return DEFAULT_CONFIG
    except Exception as e:
        logging.error(f"⚠️ Error creating config file: {e}")
        return DEFAULT_CONFIG 