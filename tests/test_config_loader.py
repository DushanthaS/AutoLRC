import os
import json
import pytest
from src.config_loader import get_api_key, validate_language, load_config, SUPPORTED_LANGUAGES

@pytest.fixture
def temp_env_file(tmp_path):
    """Create a temporary .env file."""
    env_file = tmp_path / ".env"
    env_file.write_text("GEMINI_API_KEY=test_api_key")
    return env_file

@pytest.fixture
def temp_config_file(tmp_path):
    """Create a temporary config file."""
    config_file = tmp_path / "config.json"
    config = {
        "LANGUAGE": "English",
        "USE_VOCAL_ISOLATION": True,
        "CREATE_LRC": True,
        "CREATE_TXT": True
    }
    config_file.write_text(json.dumps(config))
    return config_file

def test_get_api_key_with_env_file(temp_env_file, monkeypatch):
    """Test getting API key from .env file."""
    # Set the current directory to where the temp .env file is
    monkeypatch.chdir(temp_env_file.parent)
    
    api_key = get_api_key()
    assert api_key == "test_api_key"

def test_get_api_key_without_env_file(monkeypatch):
    """Test getting API key when .env file doesn't exist."""
    # Remove GEMINI_API_KEY from environment
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    
    api_key = get_api_key()
    assert api_key is None

def test_validate_language():
    """Test language validation."""
    # Test with full language name
    assert validate_language("English") == "en"
    assert validate_language("Sinhala") == "si"
    
    # Test with language code
    assert validate_language("en") == "en"
    assert validate_language("si") == "si"
    
    # Test with unsupported language
    assert validate_language("Invalid") == "en"

def test_load_config_with_env_file(temp_env_file, temp_config_file, monkeypatch):
    """Test loading config with .env file."""
    # Set the current directory to where the temp files are
    monkeypatch.chdir(temp_env_file.parent)
    
    config = load_config()
    assert config is not None
    assert config["LANGUAGE"] == "en"
    assert config["USE_VOCAL_ISOLATION"] is True
    assert config["CREATE_LRC"] is True
    assert config["CREATE_TXT"] is True

def test_load_config_without_env_file(monkeypatch):
    """Test loading config without .env file."""
    # Remove GEMINI_API_KEY from environment
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    
    config = load_config()
    assert config is None

def test_supported_languages():
    """Test that all supported languages are valid."""
    for lang, code in SUPPORTED_LANGUAGES.items():
        assert validate_language(lang) == code
        assert validate_language(code) == code 