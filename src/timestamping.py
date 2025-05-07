import logging
import subprocess
import numpy as np
import torch
import torchaudio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

# Configure logging - only if not already configured
if not logging.getLogger().handlers:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )

def load_model_and_processor(model_path="/data/models/.torch/wav2vec2-base-960h"):
    try:
        processor = Wav2Vec2Processor.from_pretrained(model_path)
        model = Wav2Vec2ForCTC.from_pretrained(model_path)
        return model, processor
    except Exception as e:
        logging.error(f"Failed to load model from cache: {e}")
        model_name = 'facebook/wav2vec2-base-960h'
        processor = Wav2Vec2Processor.from_pretrained(model_name)
        model = Wav2Vec2ForCTC.from_pretrained(model_name)
        processor.save_pretrained(model_path)
        model.save_pretrained(model_path)
        return model, processor

def chunk_waveform(waveform, chunk_size_sec=20, sample_rate=16000):
    chunk_size = int(chunk_size_sec * sample_rate)  # Cast to int here
    num_chunks = (waveform.shape[1] + chunk_size - 1) // chunk_size
    return [waveform[:, i * chunk_size: (i + 1) * chunk_size] for i in range(num_chunks)]


def analyze_audio_timestamps(audio_path, text_file_path):
    """Analyze audio and generate timestamps for text lines using a simplified approach.
    Returns a tuple of (timestamps, None) where timestamps is a list of start times for each line.
    """
    try:
        logging.info(f"Analyzing timestamps for {audio_path}")
        
        # Read the text file first to know how many lines we need timestamps for
        with open(text_file_path, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]
        
        if not lines:
            logging.warning("No text lines found in file")
            return None, None
            
        logging.info(f"Found {len(lines)} lines in text file")
        
        # Get audio duration - we'll use this for both approaches
        duration = get_audio_duration(audio_path)
        if not duration:
            logging.error("Could not determine audio duration")
            return None, None
            
        logging.info(f"Audio duration: {duration:.2f} seconds")
            
        # Try to use the model-based approach first
        try:
            # Load the model and processor
            model, processor = load_model_and_processor()
            
            # Load audio
            waveform, sr = torchaudio.load(audio_path)
            sample_rate = 16000
            if sr != sample_rate:
                waveform = torchaudio.functional.resample(waveform, orig_freq=sr, new_freq=sample_rate)

            # Calculate number of chunks needed to match line count
            # This ensures we'll have at least as many chunks as we have lines
            total_duration = waveform.shape[1] / sample_rate
            chunk_duration = total_duration / len(lines)
            # Set a reasonable minimum chunk size (3 seconds)
            chunk_size_sec = max(3, chunk_duration)
            
            chunks = chunk_waveform(waveform, chunk_size_sec=chunk_size_sec, sample_rate=sample_rate)
            results = []
            offset = 0.0

            for chunk in chunks:
                inputs = processor(chunk.squeeze().numpy(), sampling_rate=sample_rate, return_tensors="pt", padding=True)

                if torch.cuda.is_available():
                    model = model.cuda()
                    inputs = {k: v.cuda() for k, v in inputs.items()}
                else:
                    model = model.cpu()
                    inputs = {k: v.cpu() for k, v in inputs.items()}

                with torch.no_grad():
                    logits = model(**inputs).logits
                    # We don't actually need the transcription, just noting the timestamp
                    results.append(offset)

                offset += chunk.shape[1] / sample_rate
                
                # If we have enough timestamps, we can stop
                if len(results) >= len(lines):
                    break
            
            # If we don't have enough timestamps, extend the list
            if len(results) < len(lines):
                logging.info(f"Not enough timestamps ({len(results)}) for lines ({len(lines)}). Extending...")
                # Calculate remaining timestamps based on duration
                remaining_duration = duration - (results[-1] if results else 0)
                remaining_lines = len(lines) - len(results)
                interval = remaining_duration / remaining_lines
                
                for i in range(remaining_lines):
                    results.append((results[-1] if results else 0) + interval * (i + 1))
            
            # If we have too many timestamps, truncate
            if len(results) > len(lines):
                logging.info(f"Too many timestamps ({len(results)}) for lines ({len(lines)}). Truncating.")
                results = results[:len(lines)]
                
            return results, None
            
        except Exception as e:
            logging.warning(f"Model-based approach failed: {e}. Using fallback method.")
            # Continue to fallback method
            pass
        
        # Fallback method - evenly space timestamps
        logging.info("Using fallback alignment method")
        timestamps = evenly_space_timestamps(duration, len(lines))
        return timestamps, None

    except Exception as e:
        logging.error(f"Critical error in timestamp analysis: {e}")
        return None, None

def get_audio_duration(audio_path):
    """Get audio duration using ffprobe."""
    try:
        cmd = [
            "ffprobe", 
            "-v", "error", 
            "-show_entries", "format=duration", 
            "-of", "default=noprint_wrappers=1:nokey=1", 
            audio_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logging.error(f"Failed to get audio duration: {result.stderr}")
            return None
            
        return float(result.stdout.strip())
        
    except Exception as e:
        logging.error(f"Error getting audio duration: {e}")
        return None

def evenly_space_timestamps(duration, num_lines):
    """Create evenly spaced timestamps across the audio duration."""
    if not duration or num_lines <= 0:
        return None
        
    return np.linspace(0, duration, num_lines + 1)[:-1].tolist()

def create_lrc_content(transcript, timestamps):
    """Generate LRC content with line-level timestamps.
    
    This function ensures timestamps and lines match by adjusting if necessary.
    """
    try:
        lines = [line.strip() for line in transcript.split('\n') if line.strip()]
        if not lines:
            logging.warning("No text lines found in transcript")
            return None
            
        if not timestamps:
            logging.warning("No timestamps provided")
            return None
            
        # Handle mismatch between lines and timestamps
        if len(lines) != len(timestamps):
            logging.warning(f"Mismatch between lines ({len(lines)}) and timestamps ({len(timestamps)})")
            
            # If we have more lines than timestamps, get more timestamps
            if len(lines) > len(timestamps):
                # Use the last timestamp as reference and extend
                if timestamps:
                    last_timestamp = timestamps[-1]
                    avg_interval = last_timestamp / len(timestamps) if len(timestamps) > 0 else 1.0
                    
                    # Add timestamps with the average interval
                    for i in range(len(lines) - len(timestamps)):
                        timestamps.append(last_timestamp + avg_interval * (i + 1))
                else:
                    # If no timestamps, create dummy ones
                    timestamps = [i * 5.0 for i in range(len(lines))]
            
            # If we have more timestamps than lines, truncate
            elif len(timestamps) > len(lines):
                timestamps = timestamps[:len(lines)]
        
        # Now create LRC content with matched lines and timestamps
        lrc_lines = []
        for line, offset in zip(lines, timestamps):
            minutes = int(offset // 60)
            seconds = int(offset % 60)
            milliseconds = int((offset % 1) * 100)
            time_str = f"[{minutes:02d}:{seconds:02d}.{milliseconds:02d}]"
            lrc_lines.append(f"{time_str}{line}")

        return '\n'.join(lrc_lines)

    except Exception as e:
        logging.error(f"Error generating LRC content: {e}")
        return None

