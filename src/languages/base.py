"""
Base class for language mapping
"""
from abc import ABC, abstractmethod
from typing import List, Tuple

class BaseLanguageMapper(ABC):
    """Base class for language-specific character mapping"""
    
    @abstractmethod
    def preprocess_text(self, text: str) -> Tuple[List[str], List[str]]:
        """
        Split text into words with original and romanized versions
        
        Args:
            text: Input text in the target language
            
        Returns:
            Tuple of (original words, romanized words)
        """
        pass
    
    @abstractmethod
    def create_token_sequence(self, romanized_words: List[str], labels: List[str]) -> List[int]:
        """
        Create token sequence for alignment
        
        Args:
            romanized_words: List of romanized words
            labels: List of valid labels for the model
            
        Returns:
            List of token indices
        """
        pass 