"""
Language mapping system for forced alignment
"""

from .base import BaseLanguageMapper
from .sinhala import SinhalaMapper
from .english import EnglishMapper

LANGUAGE_MAPPERS = {
    "Sinhala": SinhalaMapper,
    "English": EnglishMapper,
}

def get_language_mapper(language: str) -> BaseLanguageMapper:
    """Get the appropriate language mapper for the specified language"""
    mapper_class = LANGUAGE_MAPPERS.get(language, EnglishMapper)
    return mapper_class() 