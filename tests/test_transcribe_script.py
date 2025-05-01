import os
import json
import pytest
import sys
from unittest.mock import patch, MagicMock, mock_open
from pathlib import Path

# Add the project root directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.transcribe_script import (
    load_config,
    save_config,
    format_lrc_time,
    get_audio_duration,
    check_demucs_installed,
    sanitize_filename,
    create_timestamped_words,
    create_lrc_file,
    create_text_file,
    cleanup_temp_files
)

# Test data
SAMPLE_CONFIG = {
    "GEMINI_API_KEY": "test_api_key"
}

def test_load_config_existing():
    """Test loading existing config file"""
    with patch('builtins.open', mock_open(read_data=json.dumps(SAMPLE_CONFIG))):
        with patch('os.path.exists', return_value=True):
            config = load_config()
            assert config == SAMPLE_CONFIG

def test_load_config_missing():
    """Test loading missing config file"""
    with patch('os.path.exists', return_value=False):
        config = load_config()
        assert config == {"GEMINI_API_KEY": ""}

def test_save_config():
    """Test saving config file"""
    mock_file = mock_open()
    with patch('builtins.open', mock_file):
        with patch('os.makedirs') as mock_makedirs:
            with patch('os.chmod') as mock_chmod:
                save_config(SAMPLE_CONFIG)
                mock_makedirs.assert_called_once()
                written_content = ''.join(call.args[0] for call in mock_file().write.call_args_list)
                expected_content = json.dumps(SAMPLE_CONFIG, indent=4)
                assert written_content == expected_content
                mock_chmod.assert_called_once()

def test_format_lrc_time():
    """Test LRC time formatting"""
    assert format_lrc_time(65.5) == "[01:05.50]"
    assert format_lrc_time(0) == "[00:00.00]"
    assert format_lrc_time(3600) == "[60:00.00]"

def test_get_audio_duration():
    """Test getting audio duration"""
    mock_info = {'duration': '120.5'}
    with patch('pydub.utils.mediainfo', return_value=mock_info):
        with patch('builtins.float', lambda x: float(x)):
            duration = get_audio_duration("test.mp3")
            assert duration == 120.5


            
def test_check_demucs_installed():
    """Test Demucs installation check"""
    with patch('subprocess.run') as mock_run:
        # Test when Demucs is installed
        mock_run.return_value.returncode = 0
        assert check_demucs_installed() is True

        # Test when Demucs is not installed
        mock_run.return_value.returncode = 1
        assert check_demucs_installed() is False

def test_sanitize_filename():
    """Test filename sanitization"""
    assert sanitize_filename("test file.mp3") == "test_file_mp3"
    assert sanitize_filename("test@#$file.mp3") == "test_file_mp3"
    assert sanitize_filename("test__file.mp3") == "test_file_mp3"

# ========================
# Processing Function Tests
# ========================
def test_create_timestamped_words():
    """Test creation of timestamped words"""
    transcription = "මම ගෙදර යන්නම්"
    duration = 3.0
    result = create_timestamped_words(transcription, duration)
    
    assert len(result) == 3  # Three words
    assert result[0]["word"] == "මම"
    assert result[0]["start"] == 0.0
    assert result[0]["duration"] == 1.0

def test_create_lrc_file():
    """Test LRC file creation"""
    timestamped_words = [
        {"word": "මම", "start": 0.0, "duration": 1.0},
        {"word": "ගෙදර", "start": 1.0, "duration": 1.0},
        {"word": "යන්නම්", "start": 2.0, "duration": 1.0}
    ]
    
    mock_file = mock_open()
    with patch('builtins.open', mock_file):
        with patch('os.makedirs') as mock_makedirs:
            create_lrc_file(timestamped_words, "test.lrc")
            mock_makedirs.assert_called_once()
            mock_file().write.assert_called()

def test_create_text_file():
    """Test text file creation"""
    transcription = "මම ගෙදර යන්නම්"
    mock_file = mock_open()
    with patch('builtins.open', mock_file):
        with patch('os.makedirs') as mock_makedirs:
            create_text_file(transcription, "test.txt")
            mock_makedirs.assert_called_once()
            mock_file().write.assert_called_once_with(transcription)

def test_cleanup_temp_files():
    """Test cleanup of temporary files"""
    with patch('os.path.exists', return_value=True):
        with patch('os.remove') as mock_remove:
            with patch('shutil.rmtree') as mock_rmtree:
                cleanup_temp_files(["test1.wav", "test2.wav"])
                assert mock_remove.call_count == 2
                assert mock_rmtree.call_count == 2

# ========================
# Integration Tests
# ========================
@pytest.mark.integration
def test_process_audio_file():
    """Test full audio file processing"""
    with patch('src.transcribe_script.isolate_vocals') as mock_isolate:
        with patch('src.transcribe_script.convert_to_wav') as mock_convert:
            with patch('src.transcribe_script.transcribe_with_gemini') as mock_transcribe:
                with patch('src.transcribe_script.create_text_file') as mock_create_text:
                    with patch('src.transcribe_script.create_lrc_file') as mock_create_lrc:
                        # Setup mocks
                        mock_isolate.return_value = "vocals.wav"
                        mock_convert.return_value = "converted.wav"
                        mock_transcribe.return_value = "මම ගෙදර යන්නම්"
                        
                        # Call the function
                        from src.transcribe_script import process_audio_file
                        result = process_audio_file(
                            "test.mp3",
                            "test_api_key",
                            "output",
                            create_lrc=True,
                            create_txt=True,
                            use_vocal_isolation=True
                        )
                        
                        assert result is True
                        mock_isolate.assert_called_once()
                        mock_convert.assert_called_once()
                        mock_transcribe.assert_called_once()
                        mock_create_text.assert_called_once()
                        mock_create_lrc.assert_called_once()

if __name__ == '__main__':
    pytest.main([__file__]) 