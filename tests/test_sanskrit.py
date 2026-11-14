#!/usr/bin/env python3
"""
Tests for the Sanskrit stack.

Run with:  python3 tests/test_sanskrit.py     (or: pytest tests/)
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from sanskrit_llm import akshara
from sanskrit_llm.akshara import N_FEATURES, aksharas, segment
from sanskrit_llm.model import ModelConfig, SanskritLM, stream_tokens
from sanskrit_llm.tokenizer import AksharaTokenizer
from sanskrit_llm.transliterate import (
    devanagari_to_iast, iast_to_devanagari, detect_script, to_devanagari,
)

GITA = "धर्मक्षेत्रे कुरुक्षेत्रे समवेता युयुत्सवः । मामकाः पाण्डवाश्चैव किमकुर्वत सञ्जय ॥"
TOK_PATH = Path("artifacts/tokenizer.json")


# ───────────────────────── transliteration ─────────────────────────

def test_iast_roundtrip():
    cases = [
        "धर्मक्षेत्रे कुरुक्षेत्रे समवेता युयुत्सवः ।",
        "श्रीमद्भगवद्गीता",
        "संस्कृतम्",
        "रामो दशरथपुत्रः",
        "ज्ञानम्",
        "ॐ नमः शिवाय",
        "तदिति",
        "अग्निमीळे पुरोहितं",
        "कॢप्तम्",
    ]
    for dev in cases:
        iast = devanagari_to_iast(dev)
        back = iast_to_devanagari(iast)
        assert back == dev, f"round-trip failed: {dev!r} -> {iast!r} -> {back!r}"


def test_iast_to_devanagari_conjuncts():
    assert iast_to_devanagari("dharmakṣetre") == "धर्मक्षेत्रे"
    assert iast_to_devanagari("saṃskṛtam") == "संस्कृतम्"
    assert iast_to_devanagari("tad iti") == "तद् इति"
    assert iast_to_devanagari("śrībhagavān uvāca") == "श्रीभगवान् उवाच"
    assert iast_to_devanagari("kṛto 'tra") == "कृतो ऽत्र"
    assert iast_to_devanagari("oṃ namaḥ") == "ॐ नमः"


def test_script_detection():
    assert detect_script("धर्म") == "devanagari"
    assert detect_script("dharma") == "latin"
    assert detect_script("dharmakṣetra") == "iast"
    assert to_devanagari("dharmakṣetre") == "धर्मक्षेत्रे"


# ───────────────────────── akṣara segmentation ─────────────────────────

def test_akshara_segmentation():
    assert aksharas("धर्मक्षेत्रे") == ["ध", "र्म", "क्षे", "त्रे"]
    assert aksharas("तद् इति") == ["त", "द्", "इ", "ति"]
    assert aksharas("संस्कृतम्") == ["सं", "स्कृ", "त", "म्"]
    assert aksharas("ज्ञानम्") == ["ज्ञा", "न", "म्"]


def test_word_final_virama_does_not_swallow_text():
    """Regression: a dangling virama once consumed the rest of the document."""
    text = "तद् इति वदति । अपरं च ॥ रामः"
    units = segment(text)
    assert len(units) < 40, f"segmentation collapsed: {len(units)} units"
    assert "".join(u.text for u in units) == text
    assert units[-1].text == "मः"          # last akṣara of "रामः"


def test_segmentation_preserves_text():
    for text in [GITA, "ॐ नमः शिवाय ॥", "abc 123 धर्म"]:
        assert "".join(u.text for u in segment(text)) == text


def test_features_shape():
    f = akshara.features("क्षे")
    assert len(f) == N_FEATURES
    assert f[0] == 1.0  # contains virama
    assert akshara.features("क")[0] == 0.0


# ───────────────────────── tokenizer ─────────────────────────

def test_tokenizer_roundtrip():
    if not TOK_PATH.exists():
        print("  (tokenizer not trained yet — skipped)")
        return
    tok = AksharaTokenizer.load(TOK_PATH)
    for text in [GITA, "रामः कः आसीत् ?", "अग्निमीळे पुरोहितं यज्ञस्य देवम् ऋत्विजम् ।"]:
        ids = tok.encode(text)
        assert tok.decode(ids) == text, f"decode mismatch for {text!r}"


def test_tokens_never_split_an_akshara():
    """Every token must be made of whole akṣaras (no orphan virama/matra)."""
    if not TOK_PATH.exists():
        return
    tok = AksharaTokenizer.load(TOK_PATH)
    for token in tok.vocab:
        if token.startswith("<"):
            continue
        rebuilt = "".join(u.text for u in segment(token))
        assert rebuilt == token, f"token {token!r} is not akṣara-aligned"
        if len(token) > 1:
            assert token[0] not in akshara.MATRAS, f"token {token!r} starts mid-syllable"
            assert token[0] != akshara.VIRAMA, f"token {token!r} starts with a virama"


def test_no_merge_across_danda():
    if not TOK_PATH.exists():
        return
    tok = AksharaTokenizer.load(TOK_PATH)
    for token in tok.vocab:
        assert "।" not in token or token == "।"
        assert "॥" not in token or token == "॥"
        assert "\n" not in token


def test_tokenizer_unknown_and_specials():
    if not TOK_PATH.exists():
        return
    tok = AksharaTokenizer.load(TOK_PATH)
    ids = tok.encode("<user> रामः <assistant>")
    assert tok.vocab[ids[0]] == "<user>"
    assert tok.vocab[ids[-1]] == "<assistant>"
    # an emoji / rare symbol must not crash
    assert isinstance(tok.encode("रामः 🙂"), list)


# ───────────────────────── model ─────────────────────────

def test_model_forward_and_cache():
    torch.manual_seed(0)
    cfg = ModelConfig(vocab_size=64, d_model=32, n_layers=2, n_heads=4, d_ff=64,
                      max_seq_len=32)
    model = SanskritLM(cfg).eval()
    feats = torch.zeros(64, N_FEATURES)
    ids = torch.randint(0, 64, (2, 10))

    with torch.no_grad():
        logits, loss, _ = model(ids, feats[ids], targets=ids)
    assert logits.shape == (2, 10, 64)
    assert torch.isfinite(loss)

    # cached decoding must equal a single full forward pass
    with torch.no_grad():
        full, _, _ = model(ids, feats[ids])
        pre, _, past = model(ids[:, :4], feats[ids[:, :4]], use_cache=True)
        outs = [pre]
        for i in range(4, 10):
            lg, _, past = model(ids[:, i:i + 1], feats[ids[:, i:i + 1]],
                                past_kv=past, use_cache=True, pos_offset=i)
            outs.append(lg)
        cached = torch.cat(outs, dim=1)
    assert torch.allclose(full, cached, atol=1e-5), "KV cache diverges from full forward"


def test_sampling_runs():
    torch.manual_seed(0)
    cfg = ModelConfig(vocab_size=64, d_model=32, n_layers=2, n_heads=4, d_ff=64,
                      max_seq_len=16)
    model = SanskritLM(cfg).eval()
    feats = torch.zeros(64, N_FEATURES)
    prompt = torch.randint(0, 64, (1, 3))
    out = list(stream_tokens(model, feats, prompt, max_new_tokens=5,
                            temperature=1.0, top_k=10))
    assert len(out) <= 5
    assert all(0 <= t < 64 for t in out)


# ───────────────────────── runner ─────────────────────────

def main() -> int:
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  ✓ {name}")
        except AssertionError as exc:
            failed += 1
            print(f"  ✗ {name}: {exc}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"  ✗ {name}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
