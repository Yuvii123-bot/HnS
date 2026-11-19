#!/usr/bin/env python3
"""Validate structural provenance and basic duplicates, not Sanskrit correctness."""
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data/ramayana_qa_authored/balakanda_intro_v1.jsonl'


def audit(source):
    raw = source.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    lines = raw.decode('utf-8').splitlines()
    records = [json.loads(x) for x in DATA.read_text(encoding='utf-8').splitlines() if x]
    ids, questions = set(), set()
    for row in records:
        assert row['id'] not in ids, f"Duplicate ID: {row['id']}"
        assert row['question'] not in questions, f"Duplicate question: {row['question']}"
        ids.add(row['id']); questions.add(row['question'])
        assert row['answer'].strip() and row['question'].strip()
        assert row['language'] == 'sa-Deva'
        assert row['status'] == 'authored_unreviewed' and row['split'] == 'quarantine'
        s = row['source']
        assert s['file_sha256'] == digest
        assert 1 <= s['line_start'] <= s['focus_line'] == s['line_end'] <= len(lines)
        assert s['evidence'] == lines[s['line_start']-1:s['line_end']]
    print(f'{len(records)} entries; unique IDs and exact questions; source hashes and context verified. '
          'Linguistic accuracy, rights, semantic overlap and held-out status NOT verified.')


if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise SystemExit('Usage: python scripts/audit_ramayana_authored.py /path/to/balakanda_sanskrit.txt')
    audit(Path(sys.argv[1]))
