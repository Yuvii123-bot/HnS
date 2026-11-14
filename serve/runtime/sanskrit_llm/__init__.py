"""Torch-free subset of sanskrit_llm, staged for serverless deployment."""
from .akshara import N_FEATURES, aksharas, features, segment
from .tokenizer import AksharaTokenizer
from .transliterate import normalize_devanagari
from .numpy_lm import NumpyLM, Runtime
__all__ = ["AksharaTokenizer", "NumpyLM", "Runtime", "N_FEATURES",
           "segment", "aksharas", "features", "normalize_devanagari"]
