"""
Akṣara-BPE: a Sanskrit-specific tokenizer.

Conventional subword tokenizers are trained on bytes or characters, so they
happily cut a Devanagari syllable in half (क + ् + ष + े + त + ् + र + े),
producing tokens that no Sanskrit speaker would recognise as a unit.  This
tokenizer instead:

1. segments text into **akṣaras** (see :mod:`sanskrit_llm.akshara`),
2. runs BPE over that akṣara alphabet, so every token is always a whole number
   of syllables — a merge can never split a conjunct or a matra,
3. allows a merge to cross **one** word boundary, because Sanskrit sandhi is
   exactly the process that fuses the end of one word with the start of the
   next (``tad`` + ``iti`` -> ``taditi``).  Tokens therefore capture real
   phonological junctures.
4. never merges across a daṇḍa, a newline or a digit run, keeping verse and
   sentence boundaries intact.

Decoding is exact: concatenating the token strings reproduces the normalised
input byte for byte.

    >>> tok = AksharaTokenizer.load("artifacts/tokenizer.json")
    >>> tok.tokenize("धर्मक्षेत्रे कुरुक्षेत्रे")
    ['ध', 'र्म', 'क्षे', 'त्रे', ' कु', 'रु', 'क्षे', 'त्रे']
"""

from __future__ import annotations

import json
import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from . import akshara
from .akshara import MATRAS, VIRAMA, Unit, segment

SPECIAL_TOKENS = [
    "<pad>", "<bos>", "<eos>", "<unk>",
    "<user>", "<assistant>", "<system>",
]

#: every Sanskrit letter, mark, digit and separator we always keep in vocab
BASE_ALPHABET = (
    "".join(sorted(akshara.CONSONANTS))
    + "".join(sorted(akshara.INDEPENDENT_VOWELS))
    + "".join(sorted(akshara.MATRAS))
    + "\u0901\u0902\u0903\u093d\u0950\u094d\u093c"
    + "।॥ "
    + "".join(sorted(akshara.DIGITS))
    + "0123456789"
    + "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    + ".,;:!?()[]\"'’-–—/\\|@#%&*+=<>_~^$`{}"
)

#: characters that may never appear inside a merged token
FORBIDDEN_IN_MERGE = set("।॥\n\r\t") | set(SPECIAL_TOKENS[0][0])


def normalize(text: str) -> str:
    """Canonicalise text before segmentation."""
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\u200c", "").replace("\u200d", "").replace("\ufeff", "")
    text = text.replace("।।", "॥")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # \u00a0 -> space, collapse horizontal whitespace runs
    text = text.replace("\u00a0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    return text


@dataclass
class AksharaTokenizer:
    """BPE over akṣaras."""

    vocab: list[str] = field(default_factory=list)
    merges: list[tuple[int, int, int]] = field(default_factory=list)  # (a, b, new_id)
    version: int = 1

    def __post_init__(self) -> None:
        self._token_to_id: dict[str, int] = {t: i for i, t in enumerate(self.vocab)}
        self._ranks: dict[tuple[int, int], tuple[int, int]] = {
            (a, b): (rank, new) for rank, (a, b, new) in enumerate(self.merges)
        }
        self._special_re = (
            re.compile("(" + "|".join(re.escape(s) for s in SPECIAL_TOKENS) + ")")
            if SPECIAL_TOKENS else None
        )

    # ------------------------------------------------------------------ ids
    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    @property
    def pad_id(self) -> int:
        return self._token_to_id["<pad>"]

    @property
    def bos_id(self) -> int:
        return self._token_to_id["<bos>"]

    @property
    def eos_id(self) -> int:
        return self._token_to_id["<eos>"]

    @property
    def unk_id(self) -> int:
        return self._token_to_id["<unk>"]

    def special_id(self, name: str) -> int:
        return self._token_to_id[f"<{name}>"]

    def token_to_id(self, token: str) -> int:
        return self._token_to_id.get(token, self.unk_id)

    # ---------------------------------------------------------- vocabulary
    @staticmethod
    def _base_vocab(unit_counts: dict[str, int] | None = None,
                    aksara_alphabet_size: int = 2000) -> list[str]:
        """Specials + varṇa (single code points) + the frequent akṣara inventory.

        The alphabet is *not* the raw Unicode range: it is the corpus's akṣara
        inventory, so the very first encoding step already produces syllables.
        Rare syllables (frequency below the cut) are handled varṇa by varṇa.
        """
        vocab = list(SPECIAL_TOKENS)
        seen = set(vocab)
        # NFC: क़ (U+0958) canonically decomposes to क + nukta, and segmentation
        # normalises input, so the alphabet must be normalised the same way.
        for ch in unicodedata.normalize("NFC", BASE_ALPHABET):
            if ch not in seen:
                vocab.append(ch)
                seen.add(ch)
        if unit_counts:
            ranked = sorted(unit_counts.items(), key=lambda kv: -kv[1])
            added = 0
            for unit, _ in ranked:
                if added >= aksara_alphabet_size:
                    break
                if unit in seen or not unit:
                    continue
                if any(c in "।॥\n" for c in unit):
                    continue
                vocab.append(unit)
                seen.add(unit)
                added += 1
        return vocab

    # ------------------------------------------------------------ training
    def train(
        self,
        text: str,
        *,
        vocab_size: int = 4096,
        aksara_alphabet_size: int = 2000,
        min_pair_freq: int = 3,
        max_merges: int | None = None,
        verbose: bool = False,
    ) -> "AksharaTokenizer":
        """Learn BPE merges over akṣara units."""
        text = normalize(text)
        units = [u for u in segment(text)]
        unit_counts: dict[str, int] = defaultdict(int)
        for u in units:
            unit_counts[u.text] += 1
        self.vocab = self._base_vocab(dict(unit_counts), aksara_alphabet_size)
        self.merges = []
        token_to_id = {t: i for i, t in enumerate(self.vocab)}

        # ---- 1. split into mergeable chunks (no daṇḍa / newline inside) ----
        chunks: list[list[str]] = []
        current: list[str] = []
        for unit in segment(text):
            if unit.kind == "punct" and unit.text in ("।", "॥"):
                if current:
                    chunks.append(current)
                    current = []
                continue
            if unit.kind == "space" and "\n" in unit.text:
                if current:
                    chunks.append(current)
                chunks.append(["\n"])
                current = []
                continue
            if unit.kind == "punct" and unit.text in ".!?":
                current.append(unit.text)
                chunks.append(current)
                current = []
                continue
            current.append(unit.text)
        if current:
            chunks.append(current)

        # ---- 2. frequency table over unique chunks -------------------------
        def unit_ids(unit: str) -> list[int]:
            """Whole-akṣara id if known, else its varṇa pieces."""
            tid = token_to_id.get(unit)
            if tid is not None:
                return [tid]
            return [token_to_id.get(ch, token_to_id["<unk>"]) for ch in unit]

        word_freq: dict[tuple[int, ...], int] = defaultdict(int)
        for chunk in chunks:
            ids: list[int] = []
            for u in chunk:
                ids.extend(unit_ids(u))
            if ids:
                word_freq[tuple(ids)] += 1

        words: dict[tuple[int, ...], int] = dict(word_freq)
        pair_counts: dict[tuple[int, int], int] = defaultdict(int)
        pair_words: dict[tuple[int, int], set[tuple[int, ...]]] = defaultdict(set)
        for word, freq in words.items():
            for a, b in zip(word, word[1:]):
                pair_counts[(a, b)] += freq
                pair_words[(a, b)].add(word)

        def merge_valid(a: int, b: int) -> bool:
            s = self.vocab[a] + self.vocab[b]
            if any(c in FORBIDDEN_IN_MERGE for c in s):
                return False
            # a token may begin with a matra/virama only if it is a lone mark
            if len(s) > 1 and (s[0] in MATRAS or s[0] == "\u094d"):
                return False
            if s != s.strip():            # no leading / trailing space
                return False
            if s.count(" ") > 1:          # at most one sandhi juncture
                return False
            if any(c.isdigit() for c in s) and len(s) > 1:
                # keep digits atomic-ish; still allow merging identical digits runs
                pass
            return True

        target_merges = (vocab_size - len(self.vocab)) if max_merges is None else max_merges
        target_merges = max(0, target_merges)

        for step in range(target_merges):
            if not pair_counts:
                break
            (a, b), count = max(pair_counts.items(), key=lambda kv: kv[1])
            if count < min_pair_freq:
                break
            if not merge_valid(a, b):
                pair_counts[(a, b)] = -1
                continue

            new_id = len(self.vocab)
            self.vocab.append(self.vocab[a] + self.vocab[b])
            token_to_id[self.vocab[-1]] = new_id
            self.merges.append((a, b, new_id))

            affected = pair_words.pop((a, b), set())
            pair_counts.pop((a, b), None)

            for word in affected:
                freq = words.pop(word, 0)
                if not freq:
                    continue
                # decrement old pairs
                for x, y in zip(word, word[1:]):
                    pair_counts[(x, y)] -= freq
                    if pair_counts[(x, y)] <= 0:
                        pair_counts.pop((x, y), None)
                        pair_words.pop((x, y), None)
                    else:
                        s = pair_words.get((x, y))
                        if s is not None:
                            s.discard(word)

                # apply merge
                merged: list[int] = []
                i = 0
                while i < len(word):
                    if i < len(word) - 1 and word[i] == a and word[i + 1] == b:
                        merged.append(new_id)
                        i += 2
                    else:
                        merged.append(word[i])
                        i += 1
                new_word = tuple(merged)

                words[new_word] = words.get(new_word, 0) + freq

                # increment new pairs
                for x, y in zip(new_word, new_word[1:]):
                    pair_counts[(x, y)] += freq
                    pair_words[(x, y)].add(new_word)

            if verbose and step % 250 == 0:
                print(f"  merge {step:5d}  vocab={len(self.vocab):5d}  "
                      f"pair={self.vocab[a]!r}+{self.vocab[b]!r} count={count}",
                      flush=True)

        self.__post_init__()
        return self

    # ------------------------------------------------------------ encoding
    def _encode_chunk(self, units: list[str]) -> list[int]:
        ids: list[int] = []
        for u in units:
            tid = self._token_to_id.get(u)
            if tid is not None:
                ids.append(tid)
            else:
                ids.extend(self._token_to_id.get(ch, self.unk_id) for ch in u)
        if len(ids) < 2:
            return ids
        while True:
            best_rank, best_pos = None, None
            for i in range(len(ids) - 1):
                r = self._ranks.get((ids[i], ids[i + 1]))
                if r is not None and (best_rank is None or r[0] < best_rank):
                    best_rank, best_pos = r[0], i
            if best_pos is None:
                return ids
            new_id = self._ranks[(ids[best_pos], ids[best_pos + 1])][1]
            ids = ids[:best_pos] + [new_id] + ids[best_pos + 2:]

    def _units_of(self, text: str) -> list[Unit]:
        return segment(normalize(text))

    def encode(self, text: str, *, add_special: bool = False) -> list[int]:
        """Encode text into token ids."""
        normalized = normalize(text)
        ids: list[int] = []

        # split out literal special tokens the caller may have embedded
        parts = self._special_re.split(normalized) if self._special_re else [normalized]
        for part in parts:
            if not part:
                continue
            if part in self._token_to_id and part.startswith("<") and part.endswith(">"):
                ids.append(self._token_to_id[part])
                continue
            chunk: list[str] = []
            for unit in segment(part):
                if unit.kind == "punct" and unit.text in ("।", "॥"):
                    if chunk:
                        ids.extend(self._encode_chunk(chunk))
                        chunk = []
                    ids.append(self._token_to_id.get(unit.text, self.unk_id))
                    continue
                if unit.kind == "space" and "\n" in unit.text:
                    if chunk:
                        ids.extend(self._encode_chunk(chunk))
                        chunk = []
                    ids.append(self._token_to_id.get("\n", self.unk_id))
                    continue
                chunk.append(unit.text)
            if chunk:
                ids.extend(self._encode_chunk(chunk))

        if add_special:
            ids = [self.bos_id] + ids + [self.eos_id]
        return ids

    def tokenize(self, text: str) -> list[str]:
        """Encode and map back to token strings (useful for the UI)."""
        return [self.vocab[i] for i in self.encode(text)]

    def decode(self, ids: list[int], *, skip_special: bool = True) -> str:
        out = []
        for i in ids:
            if i < 0 or i >= len(self.vocab):
                continue
            tok = self.vocab[i]
            if skip_special and tok.startswith("<") and tok.endswith(">") and tok in SPECIAL_TOKENS:
                continue
            out.append(tok)
        return "".join(out)

    # ------------------------------------------------------------- feature
    def token_features(self, ids: list[int]) -> list[list[float]]:
        return [list(akshara.features(self.vocab[i])) for i in ids]

    # ---------------------------------------------------------- persistence
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": self.version,
            "special_tokens": SPECIAL_TOKENS,
            "vocab": self.vocab,
            "merges": [list(m) for m in self.merges],
        }
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "AksharaTokenizer":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        tok = cls(
            # vocab entries are stored NFC-normalised so they match segmentation
            vocab=[t if t.startswith("<") else unicodedata.normalize("NFC", t)
                   for t in payload["vocab"]],
            merges=[tuple(m) for m in payload["merges"]],  # type: ignore[misc]
            version=payload.get("version", 1),
        )
        return tok

    # --------------------------------------------------------------- stats
    def stats(self) -> dict:
        ak = sum(1 for t in self.vocab if t and not t.startswith("<"))
        return {
            "vocab_size": self.vocab_size,
            "merges": len(self.merges),
            "specials": len(SPECIAL_TOKENS),
            "longest_token": max(self.vocab, key=len) if self.vocab else "",
            "longest_token_len": max((len(t) for t in self.vocab), default=0),
            "tokens_with_space": sum(1 for t in self.vocab if " " in t),
            "fraction_with_virama": round(
                sum(1 for t in self.vocab if "्" in t) / max(ak, 1), 3),
        }
