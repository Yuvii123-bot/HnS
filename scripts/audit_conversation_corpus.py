#!/usr/bin/env python3
"""Audit generated curriculum files without changing them or training a model.

Exact matches are evidence of leakage, not a sufficient proof of independence.
Only use on corpora one is authorized to inspect. Outputs aggregate statistics;
no training rows or proposed evaluation answers are written to the report.
"""
import argparse
import json
from collections import Counter
from pathlib import Path

TIME_PAIRS = {
    'सप्ताहे कति दिनानि सन्ति?': 'सप्त दिनानि।',
    'वर्षे कति मासाः सन्ति?': 'द्वादश मासाः।',
    'वर्षे कति ऋतवः सन्ति?': 'षट् ऋतवः।',
    'एकस्मिन् वासरे कति होराः भवन्ति?': 'चतुर्विंशतिः होराः।',
}


def audit(corpus: Path, parts: list[Path]) -> dict:
    with (corpus / 'val.txt').open(encoding='utf-8') as source:
        validation = {line.strip() for line in source if line.strip()}
    overlap = set()
    counts = Counter()
    time = Counter()
    for path in parts:
        with path.open(encoding='utf-8') as source:
            for raw in source:
                row = raw.strip()
                if not row:
                    continue
                counts['rows'] += 1
                overlap.add(row) if row in validation else None
                user, assistant = '<user>' in row, '<assistant>' in row
                counts['paired_tags'] += int(user and assistant)
                counts['no_user_tag'] += int(not user)
                counts['no_assistant_tag'] += int(not assistant)
                counts['contains_shabdasya'] += int('शब्दस्य' in row)
                for question, correct in TIME_PAIRS.items():
                    prefix = f'<user> {question} <assistant> '
                    if row.startswith(prefix):
                        time['total'] += 1
                        time['correct'] += int(row[len(prefix):] == correct)
                        time['incorrect'] += int(row[len(prefix):] != correct)
    return {'files': [p.name for p in parts], 'counts': dict(counts),
            'validation_unique_rows': len(validation),
            'validation_unique_exact_overlaps': len(overlap),
            'fixed_time_rows': dict(time),
            'limitation': 'Exact row matching only; paraphrase, source and semantic leakage require separate review.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--corpus', required=True, type=Path)
    parser.add_argument('--part', action='append', help='part filename; default all parts')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    parts = [args.corpus / p for p in args.part] if args.part else sorted(args.corpus.glob('part_*.txt'))
    if not parts:
        parser.error('no training parts found')
    result = json.dumps(audit(args.corpus, parts), ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(result + '\n', encoding='utf-8')
    else:
        print(result)


if __name__ == '__main__':
    main()
