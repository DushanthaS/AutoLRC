#!/usr/bin/env python3
"""
Multilingual Forced Alignment with Accurate Timestamps
"""

import os
import torch
import torchaudio
from torchaudio.pipelines import WAV2VEC2_ASR_BASE_960H
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import logging
from languages import get_language_mapper

logger = logging.getLogger(__name__)

@dataclass
class WordAlignment:
    """Dataclass to store word alignment results"""
    word: str
    start_time: float
    end_time: float

class ForcedAligner:
    def __init__(self, language: str = "English"):
        """Initialize forced aligner with specified language"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.bundle = WAV2VEC2_ASR_BASE_960H
        self.model = self.bundle.get_model().to(self.device)
        self.model.eval()
        self.sample_rate = self.bundle.sample_rate
        self.labels = self.bundle.get_labels()
        self.language_mapper = get_language_mapper(language)
        self.blank = self.labels.index("_") if "_" in self.labels else 0

    def load_audio(self, path: str) -> torch.Tensor:
        """Load and preprocess audio file"""
        try:
            waveform, sample_rate = torchaudio.load(path)
            if waveform.shape[0] > 1:  # Convert stereo to mono
                waveform = waveform.mean(dim=0, keepdim=True)
            if sample_rate != self.sample_rate:  # Resample if needed
                waveform = torchaudio.functional.resample(
                    waveform, 
                    orig_freq=sample_rate, 
                    new_freq=self.sample_rate
                )
            return waveform
        except Exception as e:
            logger.error(f"Failed to load audio: {e}")
            raise

    def perform_alignment(
        self, 
        waveform: torch.Tensor, 
        tokens: List[int]
    ) -> Tuple[torch.Tensor, List[Tuple[int, int]]]:
        """Run the alignment process using Viterbi algorithm"""
        with torch.inference_mode():
            emissions, _ = self.model(waveform.to(self.device))
            emissions = emissions.squeeze(0).cpu()
        
        # Initialize trellis matrix
        num_frames, num_tokens = emissions.size(0), len(tokens)
        trellis = torch.full((num_frames + 1, num_tokens + 1), -float('inf'))
        trellis[0, 0] = 0
        trellis[1:, 0] = torch.cumsum(emissions[:, self.blank], 0)
        
        # Fill trellis
        for t in range(num_frames):
            for i in range(num_tokens):
                trellis[t+1, i+1] = max(
                    trellis[t, i+1] + emissions[t, self.blank],
                    trellis[t, i] + emissions[t, tokens[i]]
                )
        
        # Backtrack to find optimal path
        path = []
        t, i = num_frames - 1, num_tokens
        
        while t >= 0 and i >= 0:
            if i > 0 and trellis[t, i] <= (trellis[t, i-1] + emissions[t, tokens[i-1]]):
                path.append((t, i-1))
                i -= 1
            else:
                path.append((t, i))
                t -= 1
        
        return emissions, path[::-1]

    def create_word_alignment(
        self,
        original_words: List[str],
        romanized_words: List[str],
        emissions: torch.Tensor,
        path: List[Tuple[int, int]],
        waveform: torch.Tensor
    ) -> List[WordAlignment]:
        """Create word-level alignment results with accurate timing"""
        # Calculate frame duration in seconds
        total_frames = emissions.size(0)
        total_samples = waveform.size(1)
        frame_duration = (total_samples / total_frames) / self.sample_rate
        
        # Find word boundaries in token sequence
        word_indices = []
        token_pos = 0
        
        for word in romanized_words:
            if word.strip():  # Only process non-whitespace words
                start_pos = token_pos
                for char in word:
                    if char in self.labels:
                        token_pos += 1
                end_pos = token_pos
                word_indices.append((start_pos, end_pos))
                token_pos += 1  # Skip separator
        
        # Map path to words
        word_segments = []
        word_idx = 0
        original_non_ws = [w for w in original_words if w.strip()]
        
        for start_idx, end_idx in word_indices:
            if word_idx >= len(original_non_ws):
                break
                
            # Find all path entries for this word
            word_path = [p for p in path if start_idx <= p[1] <= end_idx]
            if word_path:
                start_frame = word_path[0][0]
                end_frame = word_path[-1][0]
                
                # Convert frames to seconds
                start_time = start_frame * frame_duration
                end_time = (end_frame + 1) * frame_duration  # Include full frame
                
                word_segments.append(WordAlignment(
                    word=original_non_ws[word_idx],
                    start_time=start_time,
                    end_time=end_time
                ))
                word_idx += 1
        
        return word_segments

    def align(
        self, 
        audio_path: str, 
        transcript: str
    ) -> Tuple[Optional[str], Optional[str], List[WordAlignment]]:
        """
        Perform forced alignment and return both LRC and eLRC formats
        
        Returns:
            Tuple of (lrc_content, elrc_content, word_alignments)
        """
        try:
            # Load and preprocess audio
            waveform = self.load_audio(audio_path)
            
            # Preprocess text using language mapper
            original_words, romanized_words = self.language_mapper.preprocess_text(transcript)
            
            # Create token sequence
            tokens = self.language_mapper.create_token_sequence(romanized_words, self.labels)
            
            if not tokens:
                logger.error("No valid tokens found for alignment")
                return None, None, []
            
            # Perform alignment
            emissions, path = self.perform_alignment(waveform, tokens)
            
            # Create word alignment with accurate timing
            word_segments = self.create_word_alignment(
                original_words, romanized_words, emissions, path, waveform
            )
            
            # Generate both LRC and eLRC formats
            lrc_content = self.to_lrc(word_segments)
            elrc_content = self.to_elrc(word_segments)
            
            return lrc_content, elrc_content, word_segments
            
        except Exception as e:
            logger.error(f"Alignment failed: {e}", exc_info=True)
            return None, None, []

    def to_lrc(self, alignments: List[WordAlignment]) -> str:
        """Generate standard LRC format with line-level timestamps"""
        if not alignments:
            return ""
            
        lines = []
        current_line = []
        words_per_line = 4  # Adjust based on preference
        
        for alignment in alignments:
            current_line.append(alignment)
            if len(current_line) >= words_per_line:
                lines.append(self._format_lrc_line(current_line))
                current_line = []
        
        if current_line:
            lines.append(self._format_lrc_line(current_line))
            
        return "\n".join(lines)

    def to_elrc(self, alignments: List[WordAlignment]) -> str:
        """Generate enhanced LRC format with word-level timestamps"""
        if not alignments:
            return ""
            
        lines = []
        current_line = []
        words_per_line = 3  # Fewer words for better readability
        
        for alignment in alignments:
            current_line.append(alignment)
            if len(current_line) >= words_per_line:
                lines.append(self._format_elrc_line(current_line))
                current_line = []
        
        if current_line:
            lines.append(self._format_elrc_line(current_line))
            
        return "\n".join(lines)

    def _format_lrc_line(self, alignments: List[WordAlignment]) -> str:
        """Format a single LRC line"""
        line_text = " ".join(a.word for a in alignments)
        timestamp = self._format_time(alignments[0].start_time)
        return f"{timestamp}{line_text}"

    def _format_elrc_line(self, alignments: List[WordAlignment]) -> str:
        """Format a single eLRC line with nested timestamps"""
        line_start = self._format_time(alignments[0].start_time)
        word_timestamps = [f"<{self._format_time(a.start_time)}>{a.word}" for a in alignments]
        return f"{line_start}{''.join(word_timestamps)}"

    def _format_time(self, seconds: float) -> str:
        """Format seconds to [mm:ss.xx] with proper rounding"""
        # Round to centiseconds to avoid floating point precision issues
        rounded_seconds = round(seconds, 2)
        mins = int(rounded_seconds // 60)
        secs = rounded_seconds % 60
        # Ensure two digits after decimal point
        return f"[{mins:02d}:{secs:05.2f}]"

def get_alignment(
    audio_path: str, 
    transcript: str, 
    language: str = "English"
) -> Tuple[Optional[str], Optional[str], List[WordAlignment]]:
    """
    Convenience function to get forced alignment results
    
    Returns:
        Tuple of (lrc_content, elrc_content, word_alignments)
    """
    aligner = ForcedAligner(language=language)
    return aligner.align(audio_path, transcript)