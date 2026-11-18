#!/usr/bin/env python3
"""Extract five CC0 beginner storybooks as *unreviewed*, non-training candidates.

Input repository is https://github.com/recursivepaws/amarahasa at commit
5ad695d01a2faf6c00da1aae2f71052aaf44b43f. The original content is by
Amarahasa (https://en.amarahasa.com/series/sanjaya/). Run with PyMuPDF 2.1:
  python scripts/extract_beginner_books.py --repo /path/to/amarahasa --out output
The output is not suitable for direct training: PDF text extraction can corrupt
Devanagari and includes layout, numbering and English front matter.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re

import pymupdf

TITLES = [
    '01 - संजयो न सुखी.pdf',
    '02 - हरिणी शुकं पश्यति.pdf',
    '03 - नलिनी निद्रालुः.pdf',
    '04 - आरोह, राहुल!.pdf',
    '05 - संजयः क्रीडितुम् इच्छति.pdf',
]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = []
    for i, title in enumerate(TITLES, 1):
        pdf = args.repo / 'संस्कृतम्' / 'संजयलोकः' / title
        with pymupdf.open(pdf) as doc:
            pages = [page.get_text(sort=True) for page in doc]
        text = '\n\f\n'.join(pages)
        output = args.out / f'book-{i:02}.txt'
        output.write_text(text, encoding='utf-8')
        manifest.append({'title': title, 'pdf_sha256': sha256(pdf), 'pages': len(pages),
                         'extraction_sha256': sha256(output), 'characters': len(text),
                         'nulls': text.count('\0'), 'replacement_chars': text.count('\ufffd'),
                         'suspicious_repeated_matras': len(re.findall(r'([\u093e-\u094c])\1', text)),
                         'status': 'unreviewed; do not train on this extraction'})
    (args.out / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
