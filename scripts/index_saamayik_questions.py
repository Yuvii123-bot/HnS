#!/usr/bin/env python3
"""Index locations of potential questions in Sāmayik; never infer answers.

Only hashes and public source locations are saved; copying/licensing source
content, translation, answer correctness and split safety require review.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import unicodedata

COMMIT = 'f87d54903e9f8540bda63efeb2c203e368195523'
FILES = {
    'data/bible/bible.sa': '30d1c6cff42093992f013b45c4acd58c45415d248b0983bcbeaa3cf2feafc266',
    'data/gitasopanam/gitasopanam.sa': 'a69c5a9ac07192a5553af64eeaf09ce2fef54a50b73f834418fb14147900697f',
    'data/nios/nios.sa': 'fdd19f66da2e70c419a0da0026e9f48182a5608195aa7629088332f1e16bf930',
    'data/spoken-tutorials/spoken.sa': 'df0092bedaa282ff47ccd661f0e4747816b616c7c1a0b35bb86838aea36149b3',
}


def text_hash(text):
    return hashlib.sha256(' '.join(unicodedata.normalize('NFC', text).split()).encode('utf-8')).hexdigest()


def index(source):
    rows = []
    for name, sha in FILES.items():
        data = (source / name).read_bytes()
        if hashlib.sha256(data).hexdigest() != sha:
            raise ValueError(f'checksum mismatch: {name}')
        for line_number, line in enumerate(data.decode('utf-8').splitlines(), 1):
            if '?' not in line and '？' not in line:
                continue
            rows.append({'source_path': name, 'line': line_number,
                         'url': f'https://github.com/ayushbits/saamayik/blob/{COMMIT}/{name}#L{line_number}',
                         'normalised_text_sha256': text_hash(line),
                         'question_is_entire_line': None, 'answer': None,
                         'review_status': 'unreviewed_question_locator_only',
                         'usage': 'not_training_or_heldout_evaluation'})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    rows = index(args.source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows), encoding='utf-8')
    print(json.dumps({'locators': len(rows),
                      'unique_normalised_lines': len({r['normalised_text_sha256'] for r in rows}),
                      'by_source': dict(Counter(r['source_path'] for r in rows)),
                      'qa_pairs': 0}, indent=2))


if __name__ == '__main__':
    main()
