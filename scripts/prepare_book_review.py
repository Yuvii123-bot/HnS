#!/usr/bin/env python3
"""Extract numbered Sanskrit passages for review; never produces training data.

Every output row requires source comparison and expert review. This tool does
not infer answers, alter questionable spellings, or split data into train/test.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import unicodedata

ITEM = re.compile(r'(?<!\S)([0-9]{1,3})\s+')
DEV = re.compile(r'[\u0900-\u097f]')
REPEATED_MATRA = re.compile(r'([\u093e-\u094c])\1')
LATIN = re.compile(r'[A-Za-z]')


def candidates(text, book):
    """Yield numbered segments without pairing questions and answers."""
    for line_number, line in enumerate(text.splitlines(), 1):
        matches = list(ITEM.finditer(line))
        for index, match in enumerate(matches):
            end = matches[index + 1].start() if index + 1 < len(matches) else len(line)
            piece = unicodedata.normalize('NFC', ' '.join(line[match.end():end].split()))
            if not piece or not DEV.search(piece):
                continue
            flags = []
            if REPEATED_MATRA.search(piece):
                flags.append('repeated_matra')
            if LATIN.search(piece):
                flags.append('latin_text')
            if '\ufffd' in piece or '\x00' in piece:
                flags.append('invalid_glyph')
            if not re.search(r'[।॥?!][”"\']?$', piece):
                flags.append('possibly_incomplete')
            yield {'id': f'{book}:{line_number}:{index + 1}', 'book': book,
                   'source_line': line_number, 'source_number': int(match.group(1)),
                   'text': piece, 'flags': flags, 'review_status': 'unreviewed'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    paths = sorted(args.input.glob('book-[0-9][0-9].txt'))
    if len(paths) != 5:
        parser.error(f'expected five books, found {len(paths)}')
    rows = []
    report = {'books': [], 'warning': 'All passages are unreviewed, even those with no automatic flags. Not training data.'}
    for path in paths:
        data = path.read_bytes()
        book_rows = list(candidates(data.decode('utf-8'), path.stem))
        counts = Counter(flag for row in book_rows for flag in row['flags'])
        report['books'].append({'book': path.stem, 'input_sha256': hashlib.sha256(data).hexdigest(),
                                'passages': len(book_rows), 'automatically_flagged': sum(bool(r['flags']) for r in book_rows),
                                'flags': dict(counts)})
        rows.extend(book_rows)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'review.jsonl').write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows), encoding='utf-8')
    (args.output / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
