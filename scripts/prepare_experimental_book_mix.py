#!/usr/bin/env python3
"""Prepare a bounded experimental book mix; not a verified training dataset.

Uses original Amarahasa sentence order within chapters and mixes in existing
explicit grammar instructions to reduce (not guarantee against) forgetting.
Neither the sources nor the rehearsal set have passed expert review.
"""
import json
from pathlib import Path
import random
import re

ROOT = Path(__file__).resolve().parents[1]
BOOK_DIR = ROOT / 'data/book_candidates/amarahasa_original'
OUT = ROOT / 'data/book_experiment'
GRAMMAR = re.compile(r'शब्दस्य|विभक्ति|धातोः|धातुः|लिङ्ग|लकार')


def main():
    rng = random.Random(20261118)
    manifest = json.loads((BOOK_DIR / 'manifest.json').read_text(encoding='utf-8'))
    if len(manifest['books']) != 5:
        raise ValueError('five book sources required')
    chunks = []
    for book in manifest['books']:
        path = BOOK_DIR / (book['book'] + '.jsonl')
        import hashlib
        if hashlib.sha256(path.read_bytes()).hexdigest() != book['output_sha256']:
            raise ValueError(f'checksum mismatch: {path}')
        chapter_rows = {}
        for line in path.read_text(encoding='utf-8').splitlines():
            item = json.loads(line)
            chapter = item['id'].rsplit(':', 1)[0]
            text = item['devanagari_candidate'].strip()
            if not text or re.search(r'[A-Za-z\x00\ufffd]', text):
                raise ValueError(f'bad transliteration: {item["id"]}')
            chapter_rows.setdefault(chapter, []).append(text.replace('.', '।'))
        for rows in chapter_rows.values():
            for i in range(0, len(rows), 8):
                chunks.append(' '.join(rows[i:i+8]))
    replay = [line.strip() for line in (ROOT / 'data/instruct_mix2/train.txt').read_text(encoding='utf-8').splitlines()
              if '<user>' in line and '<assistant>' in line and GRAMMAR.search(line)]
    if not replay or not chunks:
        raise ValueError('missing book or grammar content')
    desired = sum(len(row) for row in chunks)
    sampled = []
    replay_chars = 0
    while replay_chars < desired:
        row = rng.choice(replay)
        sampled.append(row)
        replay_chars += len(row)
    mix = chunks + sampled
    rng.shuffle(mix)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'train.txt').write_text('\n'.join(mix) + '\n', encoding='utf-8')
    # Required by existing framework but validation is disabled in this run;
    # this copy must NOT be represented as held out.
    (OUT / 'val.txt').write_text('\n'.join(mix[:30]) + '\n', encoding='utf-8')
    stats = {'book_chunks': len(chunks), 'book_characters': desired,
             'grammar_replay_rows': len(sampled), 'grammar_characters': replay_chars,
             'seed': 20261118, 'held_out_evaluation': False,
             'warning': 'Experimental unreviewed material; grammar retention unverified.'}
    (OUT / 'mix_manifest.json').write_text(json.dumps(stats, indent=2, ensure_ascii=False))
    print(json.dumps(stats, ensure_ascii=False))


if __name__ == '__main__':
    main()
