#!/usr/bin/env python3
"""Audit transliteration round trips, duplicates, and input hashes of story text.

Round-trip equality is not independent proof that Sanskrit is correct, that
sources are licensed, or that text is safe training data.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sanskrit_llm.transliterate import devanagari_to_iast  # noqa: E402


def normalize(value):
    return re.sub(r'\s+', ' ', value).strip().replace('ṁ', 'ṃ').replace('’', "'").lower()


def audit(root):
    manifest = json.loads((root / 'manifest.json').read_text(encoding='utf-8'))
    texts = Counter()
    bad = []
    total = questions = 0
    for book in manifest['books']:
        path = root / (book['book'] + '.jsonl')
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != book['output_sha256']:
            raise ValueError(f'checksum mismatch: {path}')
        rows = [json.loads(line) for line in raw.decode('utf-8').splitlines() if line.strip()]
        if len(rows) != book['sentences']:
            raise ValueError(f'sentence count mismatch: {path}')
        for item in rows:
            total += 1
            text = item['devanagari_candidate']
            texts[text] += 1
            questions += int('?' in text)
            if normalize(item['iast']) != normalize(devanagari_to_iast(text)):
                bad.append(item['id'])
    return {'sentences': total, 'unique_devanagari_strings': len(texts),
            'duplicate_occurrences': total - len(texts), 'sentences_with_question_mark': questions,
            'roundtrip_mismatches': bad,
            'limitations': 'Round-trip equality is not independent linguistic verification or a held-out audit.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = json.dumps(audit(args.input), ensure_ascii=False, indent=2) + '\n'
    if args.output:
        args.output.write_text(result, encoding='utf-8')
    else:
        print(result, end='')


if __name__ == '__main__':
    main()
