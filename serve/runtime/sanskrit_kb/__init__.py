"""Grammar answered from the vyākaraṇa tables, not from the weights."""
from .lexicon import lexicon, cell, decline
from .router import answer, Answer
__all__ = ["lexicon", "cell", "decline", "answer", "Answer"]
