#!/usr/bin/env python3
"""Disk-backed audit of existing QA pool, not a dataset generator.

Counts exact normalised question/answer pairs and conflicting answers without
assuming that unique questions, answers or grammar-keyword labels are correct.
The original corpus may differ from the true historical training corpus.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sqlite3
import unicodedata

GRAMMAR = re.compile(r'व्याकरण|विभक्ति|लकार|धातोः|धातुः|शब्दस्य|प्रत्यय|सन्धि|समास')


def normalise(s):
    return ' '.join(unicodedata.normalize('NFC', s).split())


def extract(line):
    if line.count('<user>') != 1 or line.count('<assistant>') != 1:
        return None
    head, rest = line.split('<user>')
    if head.strip():
        return None
    q, a = rest.split('<assistant>')
    q, a = normalise(q), normalise(a)
    if not q or not a or '<' in q or '<' in a:
        return None
    return q, a


def hash_text(s):
    return hashlib.sha256(s.encode('utf-8')).digest()


def audit(paths, db_path, report_path):
    if not paths:
        raise ValueError('no source files')
    db = sqlite3.connect(db_path)
    db.executescript('PRAGMA journal_mode=WAL; PRAGMA synchronous=NORMAL; '
                     'CREATE TABLE IF NOT EXISTS pairs (q BLOB NOT NULL, a BLOB NOT NULL, '
                     'grammar INTEGER NOT NULL, n INTEGER NOT NULL, PRIMARY KEY(q,a)) WITHOUT ROWID; '
                     'DELETE FROM pairs;')
    count = Counter()
    source_manifest = []
    insert = ('INSERT INTO pairs VALUES (?,?,?,?) ON CONFLICT(q,a) '
              'DO UPDATE SET n=n+excluded.n')
    for path in paths:
        file_counts = Counter()
        digest = hashlib.sha256()
        batch = Counter()
        with path.open('rb') as source:
            for raw in source:
                digest.update(raw)
                file_counts['lines'] += 1
                row = extract(raw.decode('utf-8', errors='strict'))
                if row is None:
                    file_counts['non_single_turn'] += 1
                    continue
                q, a = row
                is_grammar = int(bool(GRAMMAR.search(q)))
                batch[(hash_text(q), hash_text(a), is_grammar)] += 1
                file_counts['paired'] += 1
                file_counts['grammar_keyword'] += is_grammar
                if len(batch) >= 10000:
                    db.executemany(insert, ((q,a,g,n) for (q,a,g),n in batch.items()))
                    batch.clear()
        if batch:
            db.executemany(insert, ((q,a,g,n) for (q,a,g),n in batch.items()))
        db.commit()
        count.update(file_counts)
        source_manifest.append({'name': path.name, 'sha256': digest.hexdigest(), **file_counts})
        print(f'{path.name}: {file_counts["paired"]:,} paired lines', flush=True)
    summary = db.execute('SELECT count(*), coalesce(sum(n),0), '
                         'coalesce(sum(CASE WHEN grammar=1 THEN 1 ELSE 0 END),0) FROM pairs').fetchone()
    question_stats = db.execute('SELECT count(*), '
                                'coalesce(sum(CASE WHEN variants>1 THEN 1 ELSE 0 END),0) '
                                'FROM (SELECT q,count(*) variants FROM pairs GROUP BY q)').fetchone()
    report = {'counts': dict(count), 'unique_exact_qa_pairs': summary[0],
              'counted_paired_lines': summary[1], 'distinct_pairs_with_grammar_keyword': summary[2],
              'unique_normalised_questions': question_stats[0],
              'questions_with_multiple_distinct_answers': question_stats[1],
              'sources': source_manifest, 'training_ready_pairs': 0,
              'warning': 'Exact-match audit only. Multiple answers can be valid; no linguistic, semantic, licensing or near-duplicate verification. Prior training data is not held-out evaluation.'}
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    db.close()
    print(json.dumps({k:v for k,v in report.items() if k != 'sources'}, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--corpus-dir', required=True, type=Path)
    parser.add_argument('--db', required=True, type=Path)
    parser.add_argument('--report', required=True, type=Path)
    args = parser.parse_args()
    args.db.parent.mkdir(parents=True, exist_ok=True)
    audit(sorted(args.corpus_dir.glob('part_*.txt')), args.db, args.report)


if __name__ == '__main__':
    main()
