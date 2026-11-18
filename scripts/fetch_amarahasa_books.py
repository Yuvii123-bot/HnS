#!/usr/bin/env python3
"""Fetch five Amarahasa stories from their original chapters for review.

This collects source sentences, NOT training data. It preserves IAST and
chapter provenance; automated transliteration requires independent review.
No adjacent sentences are converted into question-answer pairs.
"""
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import sys
import time
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sanskrit_llm.transliterate import iast_to_devanagari  # noqa: E402

BASE = 'https://en.amarahasa.com'
BOOKS = ('sanjayo-na-sukhi', 'harini-sukam-pasyati', 'nalini-nidraluh',
         'aroha-rahula', 'sanjayah-kriditum-icchati')


class LinkParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []

    def handle_starttag(self, tag, attrs):
        if tag == 'a':
            self.links.append(dict(attrs).get('href', ''))


class SentenceParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.in_sentence = False
        self.parts = []
        self.sentences = []

    def handle_starttag(self, tag, attrs):
        if tag == 's-sent':
            if self.in_sentence:
                raise ValueError('nested source sentence')
            self.in_sentence = True
            self.parts = []

    def handle_data(self, data):
        if self.in_sentence:
            self.parts.append(data)

    def handle_endtag(self, tag):
        if tag == 's-sent' and self.in_sentence:
            sentence = ' '.join(''.join(self.parts).split())
            if not sentence:
                raise ValueError('empty source sentence')
            self.sentences.append(sentence)
            self.in_sentence = False


def source_sentences(html):
    parser = SentenceParser()
    parser.feed(html)
    if parser.in_sentence:
        raise ValueError('unterminated sentence')
    return parser.sentences


def fetch(path):
    request = Request(BASE + path, headers={'User-Agent': 'SanatanAI-source-review/1.0'})
    with urlopen(request, timeout=40) as response:
        return response.read().decode('utf-8')


def main():
    dest = ROOT / 'data/book_candidates/amarahasa_original'
    dest.mkdir(parents=True, exist_ok=True)
    summary = []
    for book in BOOKS:
        listing = fetch(f'/books/{book}/')
        links = LinkParser()
        links.feed(listing)
        pattern = re.compile(rf'^/books/{re.escape(book)}/([0-9]+)/$')
        chapter_numbers = sorted({int(m.group(1)) for link in links.links if (m := pattern.match(link))})
        if not chapter_numbers or chapter_numbers != list(range(1, max(chapter_numbers) + 1)):
            raise RuntimeError(f'{book}: missing or non-contiguous chapters: {chapter_numbers}')
        rows = []
        for number in chapter_numbers:
            path = f'/books/{book}/{number}/'
            html = fetch(path)
            sentences = source_sentences(html)
            if not sentences:
                raise RuntimeError(f'{path}: no source sentences')
            digest = hashlib.sha256(html.encode('utf-8')).hexdigest()
            for index, sentence in enumerate(sentences, 1):
                devanagari = iast_to_devanagari(sentence, om=False)
                if re.search(r'[A-Za-z]', devanagari):
                    raise RuntimeError(f'{path}:{index}: transliteration left Latin: {devanagari}')
                rows.append({'id': f'{book}:{number}:{index}', 'source_url': BASE + path,
                             'source_html_sha256': digest, 'iast': sentence,
                             'devanagari_candidate': devanagari,
                             'review_status': 'unreviewed_transliteration'})
            time.sleep(0.2)
        output = dest / f'{book}.jsonl'
        output.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows), encoding='utf-8')
        summary.append({'book': book, 'chapters': len(chapter_numbers), 'sentences': len(rows),
                        'output_sha256': hashlib.sha256(output.read_bytes()).hexdigest()})
    (dest / 'manifest.json').write_text(json.dumps({'source': BASE, 'license_note': 'Amarahasa states CC0 unless otherwise noted; verify per-source terms.',
                                                    'warning': 'Transliterated candidate text; review before training.', 'books': summary},
                                                   ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == '__main__':
    main()
