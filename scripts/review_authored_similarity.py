#!/usr/bin/env python3
"""Flag lexically similar authored questions for manual review, not automatic removal."""
import argparse
from difflib import SequenceMatcher
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.train_authored_qa import load_pairs


def find_similar(paths, threshold=0.8):
    rows = []
    for path in paths:
        for n, row in enumerate(load_pairs(path), start=2):
            rows.append({'file': str(path), 'line': n, 'question': row['question'], 'category': row['category']})
    matches = []
    for i, a in enumerate(rows):
        for b in rows[:i]:
            score = SequenceMatcher(None, a['question'], b['question'], autojunk=False).ratio()
            if score >= threshold:
                matches.append({'score': round(score, 4), 'first': b, 'second': a,
                                'status': 'unreviewed_may_be_distinct'})
    matches.sort(key=lambda r: (-r['score'], r['first']['file'], r['first']['line']))
    return matches


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch', action='append', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--threshold', type=float, default=0.8)
    args = parser.parse_args()
    if not 0 < args.threshold <= 1:
        parser.error('threshold must be between 0 and 1')
    rows = find_similar(args.batch, args.threshold)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows), encoding='utf-8')
    print(f'{len(rows)} possible lexical overlaps; not semantic verification')


if __name__ == '__main__':
    main()
