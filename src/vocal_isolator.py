import os
import shutil
import time
import subprocess
import logging
from pathlib import Path
from config_loader import TEMP_DIR, DEMUCS_OUTPUT_FOLDER
from audio_utils import sanitize_filename

def isolate_vocals(input_path, use_gpu=True, fast_mode=False):
    """Run Demucs to isolate vocals with performance optimizations
    
    Args:
        input_path (str): Path to the input audio file
        use_gpu (bool): Whether to use GPU acceleration if available
        fast_mode (bool): Use faster but slightly lower quality model
    
    Returns:
        str or None: Path to isolated vocals file, or None if process failed
    """
    logging.info("🎵 Isolating vocals with Demucs...")
    
    # Create a temporary working copy with a safe filename
    original_filename = os.path.basename(input_path)
    file_ext = os.path.splitext(original_filename)[1]
    safe_filename = sanitize_filename(os.path.splitext(original_filename)[0]) + file_ext
    
    os.makedirs(TEMP_DIR, exist_ok=True)
    
    safe_input_path = os.path.join(TEMP_DIR, safe_filename)
    logging.info(f"📝 Creating a working copy with simplified filename: {safe_filename}")
    shutil.copy2(input_path, safe_input_path)
    
    # Create output folder if needed
    output_id = f"output_{int(time.time())}"
    custom_output_dir = os.path.join(DEMUCS_OUTPUT_FOLDER, output_id)
    os.makedirs(custom_output_dir, exist_ok=True)
    
    # Convert to absolute paths with forward slashes to avoid path issues
    abs_input_path = os.path.abspath(safe_input_path).replace("\\", "/")
    abs_output_dir = os.path.abspath(custom_output_dir).replace("\\", "/")
    
    # Set up Demucs command with optimizations
    cmd = ["python3", "-m", "demucs.separate", "--two-stems=vocals"]
    
    # Select model based on speed vs quality preference
    model = "htdemucs_ft" if fast_mode else "htdemucs"
    cmd.extend(["-n", model])
    
    # Add GPU acceleration if requested and available
    if use_gpu:
        cmd.append("--device=cuda")
    
    # Add output directory and input file
    cmd.extend(["--out", abs_output_dir, abs_input_path])
    
    try:
        # Run the command
        logging.info(f"📋 Running: {' '.join(cmd)}")
        
        # Execute command with real-time output monitoring
        process = subprocess.Popen(
            cmd,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
            universal_newlines=True
        )
        
        # Monitor progress in real-time
        for line in process.stdout:
            line = line.strip()
            if "%" in line:  # This is a progress update
                logging.info(f"Progress: {line}")
            elif line:  # Only log non-empty lines
                logging.debug(line)
        
        # Wait for process to complete
        return_code = process.wait()
        
        if return_code != 0:
            logging.error(f"❌ Demucs process failed with code {return_code}")
            return None
        
        # Find the vocals file more efficiently
        base_name = os.path.splitext(os.path.basename(safe_input_path))[0]
        model_output_dir = Path(custom_output_dir) / model / base_name
        
        # Direct path to vocals file based on Demucs output pattern
        vocals_path = model_output_dir / "vocals.wav"
        
        if vocals_path.exists():
            logging.info(f"✅ Found vocals file: {vocals_path}")
            return str(vocals_path)
        
        # Fallback search if direct path failed
        for file in model_output_dir.glob("*vocal*.wav"):
            logging.info(f"✅ Found vocals file: {file}")
            return str(file)
            
        logging.error("❌ Vocals file not found after processing")
        return None
        
    except Exception as e:
        logging.error(f"❌ Vocal isolation failed: {e}", exc_info=True)
        return None
    finally:
        # Don't delete the temp file yet as we might need it for transcription
        pass