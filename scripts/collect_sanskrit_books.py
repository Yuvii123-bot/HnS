#!/usr/bin/env python3
"""Download candidate scans' OCR for review. Never start training or publish a model.

These are candidate sources, not approved training data. Retain raw OCR and
metadata unmodified and do not use without rights and linguistic review.
"""
import hashlib
import json
import re
import urllib.parse
import urllib.request
from pathlib import Path

BOOKS = [
    ('pathavali', 'Pathavali_Sanskrit_Reader_3_parts_-_T_Ganapati_Sastri1907', None),
    ('perry_primer', 'in.ernet.dli.2015.261252', None),
    ('hitopadesa_first_book', 'bub_gb_wlKV8dRl5hIC', None),
    ('lanman_reader', 'ASanskritReader', None),
    ('ballantyne_first_lessons', 'SanskritSabdaManjari', 'Sanskrit-First-Lessons-JRBallantyne-1850'),
]


def download(url):
    req = urllib.request.Request(url, headers={'User-Agent': 'SanatanAI-research/1.0'})
    with urllib.request.urlopen(req, timeout=90) as response:
        return response.read()


def main():
    root = Path('book-review')
    root.mkdir(exist_ok=True)
    report = []
    for key, identifier, startswith in BOOKS:
        metadata_url = f'https://archive.org/metadata/{identifier}'
        metadata = json.loads(download(metadata_url))
        files = [f['name'] for f in metadata.get('files', []) if f['name'].endswith('_djvu.txt')
                 and (startswith is None or f['name'].startswith(startswith))]
        if len(files) != 1:
            raise RuntimeError(f'{key}: expected exactly one OCR file; found {files!r}')
        url = f'https://archive.org/download/{identifier}/{urllib.parse.quote(files[0])}'
        data = download(url)
        text = data.decode('utf-8')
        if len(data) < 1000:
            raise RuntimeError(f'{key}: unexpectedly short OCR')
        (root / f'{key}.txt').write_bytes(data)
        (root / f'{key}.metadata.json').write_text(json.dumps(metadata.get('metadata', {}), ensure_ascii=False, indent=2))
        report.append({'key': key, 'metadata_url': metadata_url, 'ocr_url': url,
                       'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data),
                       'devanagari_characters': len(re.findall(r'[\u0900-\u097f]', text)),
                       'status': 'quarantined; rights and OCR review required'})
    (root / 'manifest.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
