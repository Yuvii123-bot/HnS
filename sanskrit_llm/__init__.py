"""Saṃskṛta-LLM: a Sanskrit-native language model stack.

Everything here is built around the akṣara (syllable): the tokenizer's alphabet,
the model's feature embeddings, and the chat interface's visualiser.

Only the torch-free parts are imported eagerly, so the tokenizer and the NumPy
inference path can be used in a serverless function without pulling in PyTorch.
"""

from .akshara import N_FEATURES, aksharas, features, segment
from .tokenizer import AksharaTokenizer
from .transliterate import (
    detect_script, devanagari_to_iast, iast_to_devanagari,
    normalize_devanagari, slp1_to_iast, iast_to_slp1, to_devanagari,
)

__all__ = [
    "AksharaTokenizer", "ModelConfig", "SanskritLM", "N_FEATURES",
    "segment", "aksharas", "features",
    "devanagari_to_iast", "iast_to_devanagari", "to_devanagari",
    "detect_script", "normalize_devanagari", "slp1_to_iast", "iast_to_slp1",
    "NumpyLM", "Runtime",
]


def __getattr__(name: str):
    """Lazy imports for the torch-backed pieces (keeps serverless cold starts small)."""
    if name in ("ModelConfig", "SanskritLM"):
        from .model import ModelConfig, SanskritLM
        return {"ModelConfig": ModelConfig, "SanskritLM": SanskritLM}[name]
    if name in ("NumpyLM", "Runtime"):
        from .numpy_lm import NumpyLM, Runtime
        return {"NumpyLM": NumpyLM, "Runtime": Runtime}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
