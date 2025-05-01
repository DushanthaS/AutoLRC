import os
import shutil
import time
import subprocess
import logging
from pathlib import Path
from config_loader import TEMP_DIR, DEMUCS_OUTPUT_FOLDER
from audio_utils import sanitize_filename

def isolate_vocals(input_path):
    """Run Demucs to isolate vocals"""
    logging.info("🎵 Isolating vocals with Demucs...")
    
    # Create a temporary working copy with a safe filename
    original_filename = os.path.basename(input_path)
    file_ext = os.path.splitext(original_filename)[1]
    safe_filename = sanitize_filename(os.path.splitext(original_filename)[0]) + file_ext
    
    os.makedirs(TEMP_DIR, exist_ok=True)
    
    safe_input_path = os.path.join(TEMP_DIR, safe_filename)
    logging.info(f"📝 Creating a working copy with simplified filename: {safe_filename}")
    shutil.copy2(input_path, safe_input_path)
    
    try:
        # Create output folder if it doesn't exist
        os.makedirs(DEMUCS_OUTPUT_FOLDER, exist_ok=True)
        
        # Create a custom output directory for this specific file to avoid conflict
        output_id = f"output_{int(time.time())}"
        custom_output_dir = os.path.join(DEMUCS_OUTPUT_FOLDER, output_id)
        os.makedirs(custom_output_dir, exist_ok=True)

        logging.info(f"🔄 Running Demucs vocal separation...")
        
        # Convert to absolute paths with forward slashes to avoid path issues
        abs_input_path = os.path.abspath(safe_input_path).replace("\\", "/")
        abs_output_dir = os.path.abspath(custom_output_dir).replace("\\", "/")
        
        # Run Demucs - explicitly specify model and stem
        cmd = [
            "python3", "-m", "demucs.separate",
            "--two-stems=vocals", 
            "-n", "htdemucs",
            "--out", abs_output_dir,
            abs_input_path
        ]
        
        logging.info(f"📋 Command: {' '.join(cmd)}")
        
        # Run the command without shell=True for more reliable execution
        result = subprocess.run(
            cmd,
            text=True, 
            capture_output=True,
            shell=False
        )

        logging.info(f"Demucs stdout: {result.stdout}")
        logging.info(f"Demucs stderr: {result.stderr}")

        if result.returncode != 0:
            logging.error(f"❌ Demucs process failed with code {result.returncode}")
            logging.warning("⚠️ Vocal isolation failed. Proceeding with original audio...")
            return None

        # Since we explicitly set the model to htdemucs, look in that folder
        base_name = os.path.splitext(os.path.basename(safe_input_path))[0]
        model_dir = "htdemucs"  # We explicitly specified this model
        
        # Try to find the vocals file with proper path handling using Path
        model_output_dir = Path(custom_output_dir) / model_dir / base_name
        if model_output_dir.exists():
            logging.info(f"🔍 Looking for vocals file in: {model_output_dir}")
            
            # Check for a vocals file with various possible names
            potential_stems = ["vocals", "vocals.wav", "vocal", "voice"]
            for stem in potential_stems:
                stem_path = model_output_dir / stem
                if stem_path.exists():
                    logging.info(f"✅ Found vocals file: {stem_path}")
                    return str(stem_path)
                
                # Also check with .wav extension if not already included
                if not stem.endswith(".wav"):
                    stem_path = model_output_dir / f"{stem}.wav"
                    if stem_path.exists():
                        logging.info(f"✅ Found vocals file: {stem_path}")
                        return str(stem_path)
            
            # If we still haven't found it, list all files in the directory
            logging.info("📋 Files found in output directory:")
            for file in model_output_dir.iterdir():
                logging.info(f"  - {file.name}")
                # If the file has vocal in the name, use it
                if "vocal" in file.name.lower() and file.name.endswith(".wav"):
                    logging.info(f"✅ Found vocals file: {file}")
                    return str(file)

        logging.error("❌ Vocals file not found after processing")
        logging.warning("⚠️ Proceeding with original audio...")
        return None

    except Exception as e:
        logging.error(f"❌ Vocal isolation failed: {e}")
        logging.warning("⚠️ Proceeding with original audio...")
        return None
    finally:
        # Don't delete the temp file yet as we might need it for transcription
        pass 