"""
Sinhala language mapping implementation
"""
import re
from typing import List, Tuple
from .base import BaseLanguageMapper

class SinhalaMapper(BaseLanguageMapper):
    """Sinhala language mapper implementation"""
    
    # Enhanced Sinhala mapping
    SINHALA_MAP = {
        'අ': 'a', 'ආ': 'aa', 'ඇ': 'ae', 'ඈ': 'aae', 'ඉ': 'i', 'ඊ': 'ii',
        'උ': 'u', 'ඌ': 'uu', 'ඍ': 'ru', 'ඎ': 'ruu', 'එ': 'e', 'ඒ': 'ee',
        'ඓ': 'ai', 'ඔ': 'o', 'ඕ': 'oo', 'ඖ': 'au',
        'ක': 'ka', 'ඛ': 'kha', 'ග': 'ga', 'ඝ': 'gha', 'ඞ': 'nga',
        'ච': 'cha', 'ඡ': 'chha', 'ජ': 'ja', 'ඣ': 'jha', 'ඤ': 'nya',
        'ට': 'ta', 'ඨ': 'tha', 'ඩ': 'da', 'ඪ': 'dha', 'ණ': 'na',
        'ත': 'tha', 'ථ': 'thha', 'ද': 'da', 'ධ': 'dha', 'න': 'na',
        'ප': 'pa', 'ඵ': 'pha', 'බ': 'ba', 'භ': 'bha', 'ම': 'ma',
        'ය': 'ya', 'ර': 'ra', 'ල': 'la', 'ව': 'va', 'ශ': 'sha',
        'ෂ': 'sha', 'ස': 'sa', 'හ': 'ha', 'ළ': 'la', 'ෆ': 'fa',
        '්': '', 'ා': 'a', 'ැ': 'e', 'ෑ': 'ee', 'ි': 'i', 'ී': 'ii',
        'ු': 'u', 'ූ': 'uu', 'ෘ': 'ru', 'ෙ': 'e', 'ේ': 'ee', 'ෛ': 'ai',
        'ො': 'o', 'ෝ': 'oo', 'ෞ': 'au', 'ං': 'ng', 'ඃ': 'h',
        ' ': ' ', '\n': ' '
    }
    
    def preprocess_text(self, text: str) -> Tuple[List[str], List[str]]:
        """Split text into words with original and romanized versions"""
        words = re.findall(r"\S+|\s", text)  # Split preserving whitespace
        original = []
        romanized = []
        
        for word in words:
            original.append(word)
            if word.strip():
                romanized.append(''.join(self.SINHALA_MAP.get(c, '') for c in word))
            else:
                romanized.append(word)
        
        return original, romanized
    
    def create_token_sequence(self, romanized_words: List[str], labels: List[str]) -> List[int]:
        """Create token sequence for alignment"""
        tokens = []
        for word in romanized_words:
            if word.strip():  # Only process non-whitespace words
                for char in word:
                    if char in labels:
                        tokens.append(labels.index(char))
                tokens.append(labels.index('|'))  # Word separator
        return tokens 