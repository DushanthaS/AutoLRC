import os
import pytest
from pathlib import Path

def pytest_configure(config):
    """Register custom marks."""
    config.addinivalue_line(
        "markers",
        "integration: mark test as an integration test"
    )

@pytest.fixture
def temp_dir(tmp_path):
    """Create a temporary directory for test files"""
    return tmp_path


@pytest.fixture
def temp_dir(tmp_path):
    """Create a temporary directory for test files"""
    return tmp_path

@pytest.fixture
def sample_audio_file(temp_dir):
    """Create a sample audio file for testing"""
    audio_path = temp_dir / "test.mp3"
    audio_path.touch()
    return str(audio_path)

@pytest.fixture
def mock_config():
    """Provide a mock configuration"""
    return {
        "GEMINI_API_KEY": "test_api_key"
    }

@pytest.fixture
def mock_audio_info():
    """Provide mock audio file information"""
    return {
        'duration': '120.5',
        'channels': '2',
        'sample_rate': '44100'
    } 
    
