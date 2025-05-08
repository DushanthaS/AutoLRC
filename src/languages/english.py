"""
English language mapping implementation
"""
import re
from typing import List, Tuple
from .base import BaseLanguageMapper

class EnglishMapper(BaseLanguageMapper):
    """English language mapper implementation"""
    
    def preprocess_text(self, text: str) -> Tuple[List[str], List[str]]:
        """Split text into words with original and romanized versions"""
        # For English, original and romanized are the same
        words = re.findall(r"\S+|\s", text)  # Split preserving whitespace
        return words, words
    
    def create_token_sequence(self, romanized_words: List[str], labels: List[str]) -> List[int]:
        """Create token sequence for alignment"""
        tokens = []
        for word in romanized_words:
            if word.strip():  # Only process non-whitespace words
                for char in word.lower():
                    if char in labels:
                        tokens.append(labels.index(char))
                tokens.append(labels.index('|'))  # Word separator
        return tokens 