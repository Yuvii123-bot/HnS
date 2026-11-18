#!/usr/bin/env python3
"""Prepare review queues, never a training or held-out evaluation set."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import unicodedata

ROOT = Path(__file__).resolve().parents[1]


def norm(s):
    return ' '.join(unicodedata.normalize('NFC', s).split())


def book_questions(book_dir):
    manifest = json.loads((book_dir / 'manifest.json').read_text(encoding='utf-8'))
    result = []
    for book in manifest['books']:
        path = book_dir / (book['book'] + '.jsonl')
        if hashlib.sha256(path.read_bytes()).hexdigest() != book['output_sha256']:
            raise ValueError(f'book checksum mismatch: {path}')
        rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line]
        for i, row in enumerate(rows):
            if '?' not in row['iast']:
                continue
            chapter = row['id'].rsplit(':', 1)[0]
            context = [r for r in rows[max(0, i-3):i+4] if r['id'].rsplit(':', 1)[0] == chapter]
            result.append({'id': row['id'], 'source_url': row['source_url'],
                           'source_html_sha256': row['source_html_sha256'],
                           'question_iast': row['iast'], 'question_candidate': row['devanagari_candidate'],
                           'context': [{'id': r['id'], 'iast': r['iast'],
                                        'devanagari_candidate': r['devanagari_candidate']} for r in context],
                           'answer': None, 'review_status': 'unreviewed',
                           'usage': 'draft_only_not_training_or_evaluation'})
    return result


def qa_candidates(catalogue):
    rows = [json.loads(x) for x in catalogue.read_text(encoding='utf-8').splitlines() if x]
    grouped = defaultdict(list)
    for row in rows:
        grouped[norm(row['question'])].append(row)
    # Do not silently choose between duplicate or potentially conflicting answers.
    result = []
    for group in grouped.values():
        if len(group) != 1:
            continue
        row = group[0]
        result.append({'id': row['source'] + ':' + row['source_id'],
                       'question': row['question'], 'answer_alternatives_raw': row['answer_raw'],
                       'choices_raw': row['choices_raw'], 'category': None,
                       'review_status': 'unreviewed', 'usage': 'draft_only_not_training_or_evaluation',
                       'review_flags': ['published_benchmark', 'rights_check', 'answer_and_category_review']
                       + (['health_content_review'] if 'ayurveda' in row['source'] else [])
                       + (['multiple_choice_context_review'] if row['choices_raw'] else [])})
    return result, len(rows) - len(result)


def write_jsonl(path, rows):
    path.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--book-dir', type=Path, default=ROOT / 'data/book_candidates/amarahasa_original')
    parser.add_argument('--catalogue', type=Path, default=ROOT / 'data/qa_100k/catalogue/sktqa_unreviewed.jsonl')
    parser.add_argument('--out', type=Path, default=ROOT / 'data/qa_25k')
    args = parser.parse_args()
    books = book_questions(args.book_dir)
    qa, excluded = qa_candidates(args.catalogue)
    args.out.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out / 'book_questions_review.jsonl', books)
    write_jsonl(args.out / 'sktqa_pairs_review.jsonl', qa)
    report = {'target_total': 25000, 'target_grammar': 250, 'target_other': 24750,
              'sktqa_qa_drafts': len(qa), 'sktqa_rows_excluded_due_to_duplicate_question': excluded,
              'book_question_occurrences_without_answers': len(books),
              'book_distinct_question_texts': len({norm(r['question_candidate']) for r in books}),
              'book_occurrences_by_title': dict(Counter(r['id'].split(':')[0] for r in books)),
              'reviewed_training_ready_qa': 0,
              'warning': 'Unreviewed source material; no training, no held-out evaluation. Books were part of prior training; context is not an answer.'}
    (args.out / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
