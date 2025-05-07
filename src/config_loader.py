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

# Supported languages and their codes
SUPPORTED_LANGUAGES = {
    "English": "en",
    "Sinhala": "si",
    "Tamil": "ta",
    "Hindi": "hi",
    "Spanish": "es",
    "French": "fr",
    "German": "de",
    "Japanese": "ja",
    "Korean": "ko",
    "Chinese": "zh",
    "Russian": "ru",
    "Arabic": "ar",
    "Portuguese": "pt",
    "Italian": "it",
    "Dutch": "nl",
    "Turkish": "tr",
    "Vietnamese": "vi",
    "Thai": "th",
    "Indonesian": "id",
    "Malay": "ms"
}

# Default configuration values
DEFAULT_CONFIG = {
    "GEMINI_MODEL": "gemini-2.0-flash-thinking",
    "LANGUAGE": "English",  # Default language for transcription
    "TEMPERATURE": 0.2,
    "TOP_P": 0.8,
    "TOP_K": 40,
    "CANDIDATE_COUNT": 1,
    "MAX_RETRIES": 3,
    "RETRY_DELAY": 5,
    "USE_VOCAL_ISOLATION": True,
    "CREATE_TXT": True,
    "CREATE_LRC": True
}

def get_api_key():
    """Get API key from .env file or environment variable."""
    # Try to get API key from .env file first
    api_key = os.getenv("GEMINI_API_KEY")
    
    if not api_key:
        logging.error("❌ GEMINI_API_KEY not found in .env file or environment variables")
        logging.error("Please create a .env file in the project root with:")
        logging.error("GEMINI_API_KEY=your_api_key_here")
        return None
    
    # Remove any quotes that might have been added
    api_key = api_key.strip('"\'')
    
    if not api_key:
        logging.error("❌ GEMINI_API_KEY is empty")
        return None
        
    logging.info("✅ GEMINI_API_KEY found")
    return api_key

def validate_language(language):
    """Validate and return the language code."""
    if language in SUPPORTED_LANGUAGES:
        return SUPPORTED_LANGUAGES[language]
    elif language in SUPPORTED_LANGUAGES.values():
        return language
    else:
        logging.warning(f"⚠️ Unsupported language: {language}. Defaulting to English.")
        return "en"

def load_config():
    """Load configuration from JSON file with defaults and environment variables."""
    # Get API key from .env file
    api_key = get_api_key()
    if not api_key:
        return None

    # Load user config if exists
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, 'r') as f:
                user_config = json.load(f)
                # Merge user config with defaults, keeping user values
                config = DEFAULT_CONFIG.copy()
                config.update(user_config)
                # Validate language
                config["LANGUAGE"] = validate_language(config["LANGUAGE"])
                return config
        except Exception as e:
            logging.error(f"⚠️ Error loading config: {e}")
            logging.warning("⚠️ Using default configuration")
            config = DEFAULT_CONFIG.copy()
            config["LANGUAGE"] = validate_language(config["LANGUAGE"])
            return config
    
    # If config file doesn't exist, create it with defaults
    try:
        os.makedirs(os.path.dirname(CONFIG_PATH), exist_ok=True)
        with open(CONFIG_PATH, 'w') as f:
            json.dump(DEFAULT_CONFIG, f, indent=4)
        os.chmod(CONFIG_PATH, 0o600)  # Secure permissions
        logging.info("✅ Created default configuration file")
        config = DEFAULT_CONFIG.copy()
        config["LANGUAGE"] = validate_language(config["LANGUAGE"])
        return config
    except Exception as e:
        logging.error(f"⚠️ Error creating config file: {e}")
        config = DEFAULT_CONFIG.copy()
        config["LANGUAGE"] = validate_language(config["LANGUAGE"])
        return config 