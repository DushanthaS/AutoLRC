import logging
import numpy as np
import librosa
import librosa.onset

def analyze_audio_timestamps(audio_path):
    """Analyze audio file to generate timestamps using librosa."""
    try:
        # Load audio file
        y, sr = librosa.load(audio_path)
        
        # Get onset strength
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        
        # Detect onset times
        onset_frames = librosa.onset.onset_detect(
            onset_envelope=onset_env,
            sr=sr,
            wait=0.1,  # Minimum time between onsets
            pre_avg=0.1,  # Pre-averaging window
            post_avg=0.1,  # Post-averaging window
            pre_max=0.1,  # Pre-maximum window
            post_max=0.1   # Post-maximum window
        )
        
        # Convert frames to timestamps
        timestamps = librosa.frames_to_time(onset_frames, sr=sr)
        logging.info(f"Generated {len(timestamps)} timestamps")
        
        return timestamps
        
    except Exception as e:
        logging.error(f"Error during audio analysis: {e}")
        return None

def create_lrc_content(transcript, timestamps):
    """Create LRC content by matching transcript lines with timestamps."""
    lrc_lines = []
    transcript_lines = [line.strip() for line in transcript.splitlines() if line.strip()]
    
    # If we have fewer timestamps than lines, we'll need to interpolate
    if len(timestamps) < len(transcript_lines):
        # Create evenly spaced timestamps
        total_duration = float(timestamps[-1]) if len(timestamps) > 0 else 0
        timestamps = np.linspace(0, total_duration, len(transcript_lines))
    
    # Match each line with a timestamp
    for i, line in enumerate(transcript_lines):
        if i < len(timestamps):
            timestamp = float(timestamps[i])
            minutes = int(timestamp / 60)
            seconds = int(timestamp % 60)
            milliseconds = int((timestamp * 100) % 100)
            lrc_line = f"[{minutes:02}:{seconds:02}.{milliseconds:02}] {line}"
            lrc_lines.append(lrc_line)
    
    return "\n".join(lrc_lines) 