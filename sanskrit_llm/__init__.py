"""Saṃskṛta-LLM: a Sanskrit-native language model stack.

Everything here is built around the akṣara (syllable): the tokenizer's alphabet,
the model's feature embeddings, and the chat interface's visualiser.
"""

from .akshara import N_FEATURES, aksharas, features, segment
from .tokenizer import AksharaTokenizer
from .transliterate import (
    detect_script, devanagari_to_iast, iast_to_devanagari,
    normalize_devanagari, slp1_to_iast, iast_to_slp1, to_devanagari,
)
from .model import ModelConfig, SanskritLM

__all__ = [
    "AksharaTokenizer", "ModelConfig", "SanskritLM", "N_FEATURES",
    "segment", "aksharas", "features",
    "devanagari_to_iast", "iast_to_devanagari", "to_devanagari",
    "detect_script", "normalize_devanagari", "slp1_to_iast", "iast_to_slp1",
]
