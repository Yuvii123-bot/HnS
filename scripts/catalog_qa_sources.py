#!/usr/bin/env python3
"""Catalogue SktQA benchmark rows for review, NOT for training.

Preserves answers and multiple-choice alternatives exactly; a semicolon can
mean an alternative answer, so never silently convert this to a chat target.
"""
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
import unicodedata

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'data/qa_100k/sources/sktqa'
OUT = ROOT / 'data/qa_100k/catalogue'
EXPECTED = {
    'sanskrit.tsv': '758e7f95969b98d7c57d0253c39549674d6776ef01eb9fb28ef87692aeb0ae24',
    'ayurveda.tsv': '4729cf14d491c8b68d7bae7cf2f7178962d84f441f31c3c791eb34db2baee2b7',
}


def normalise(text):
    return ' '.join(unicodedata.normalize('NFC', text).split())


def catalogue(source=SOURCE):
    rows = []
    for filename, expected in EXPECTED.items():
        path = source / filename
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != expected:
            raise ValueError(f'checksum mismatch: {filename}')
        with path.open(encoding='utf-8', newline='') as f:
            reader = csv.DictReader(f, delimiter='\t')
            if reader.fieldnames != ['ID', 'QUESTION', 'ANSWER', 'CHOICES']:
                raise ValueError(f'unexpected columns in {filename}')
            for record in reader:
                if None in record or not all(record[k].strip() for k in ('ID', 'QUESTION', 'ANSWER')):
                    raise ValueError(f'invalid row in {filename}: {record}')
                rows.append({'source': f'sktqa/{filename}', 'source_id': record['ID'],
                             'question': record['QUESTION'].strip(),
                             'answer_raw': record['ANSWER'].strip(),
                             'choices_raw': record['CHOICES'].strip(),
                             'category': 'unreviewed', 'review_status': 'unreviewed',
                             'usage': 'quarantine_benchmark_not_training'})
    questions = defaultdict(list)
    for row in rows:
        questions[normalise(row['question'])].append(row)
    duplicates = {k: v for k, v in questions.items() if len(v) > 1}
    conflicts = {k: v for k, v in duplicates.items()
                 if len({normalise(x['answer_raw']) for x in v}) > 1}
    report = {'total_rows': len(rows), 'source_counts': dict(Counter(x['source'] for x in rows)),
              'unique_normalised_questions': len(questions),
              'duplicate_question_groups': len(duplicates),
              'conflicting_answer_groups': len(conflicts),
              'reviewed_rows': 0, 'approved_for_training': 0,
              'target_grammar': 1000, 'target_other': 99000,
              'warning': 'Benchmark data, not a training set. Rights, answers, categories, duplication, safety and held-out partition require review.'}
    return rows, report


def main():
    rows, report = catalogue()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'sktqa_unreviewed.jsonl').write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows), encoding='utf-8')
    (OUT / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
