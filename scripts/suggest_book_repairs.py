#!/usr/bin/env python3
"""Create reversible OCR/extraction repair suggestions for human review.

Suggestions are not accepted corrections or training data. The input and
output preserve both original and proposed text, IDs, flags, and source lines.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import re

REPEATED_MATRA = re.compile(r'([\u093e-\u094c])\1+')


def suggest(item):
    original = item['text']
    proposed, edits = REPEATED_MATRA.subn(r'\1', original)
    return {**item, 'proposed_text': proposed, 'proposed_edits': edits,
            'repair_review_status': 'unreviewed'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    rows = [suggest(json.loads(line)) for line in args.input.read_text(encoding='utf-8').splitlines() if line.strip()]
    ids = [row['id'] for row in rows]
    if len(set(ids)) != len(ids):
        raise ValueError('duplicate item IDs in review input')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows), encoding='utf-8')
    by_book = Counter(row['book'] for row in rows if row['proposed_edits'])
    print(json.dumps({'passages': len(rows), 'suggestions': sum(bool(x['proposed_edits']) for x in rows),
                      'by_book': dict(by_book),
                      'status': 'All suggestions require source and linguistic review. No training data produced.'}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
