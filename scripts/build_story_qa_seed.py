#!/usr/bin/env python3
"""Build a small, manually selected story-grounded draft QA seed.

Selection checks only that nearby text appears responsive. These examples are
NOT expert-reviewed and the book lineage is already in the training history.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.prepare_25k_qa_drafts import book_questions
# Explicitly selected for an adjacent statement that appears to answer the
# question. Do not use adjacency alone to expand this list.
SELECTED = {
    'aroha-rahula:1:3', 'aroha-rahula:1:8', 'aroha-rahula:1:11',
    'aroha-rahula:1:16', 'aroha-rahula:1:20', 'aroha-rahula:1:27',
    'harini-sukam-pasyati:1:6', 'harini-sukam-pasyati:1:11',
    'harini-sukam-pasyati:1:16', 'harini-sukam-pasyati:2:5',
    'harini-sukam-pasyati:2:8', 'harini-sukam-pasyati:2:12',
    'harini-sukam-pasyati:3:2', 'nalini-nidraluh:1:5',
    'nalini-nidraluh:1:8', 'nalini-nidraluh:1:11',
    'nalini-nidraluh:2:4', 'nalini-nidraluh:2:7',
    'nalini-nidraluh:2:24', 'nalini-nidraluh:3:14',
    'sanjayah-kriditum-icchati:1:15', 'sanjayah-kriditum-icchati:1:20',
    'sanjayah-kriditum-icchati:1:23', 'sanjayah-kriditum-icchati:4:11',
    'sanjayo-na-sukhi:1:2', 'sanjayo-na-sukhi:1:5',
    'sanjayo-na-sukhi:2:5', 'sanjayo-na-sukhi:2:8',
    'sanjayo-na-sukhi:2:12', 'sanjayo-na-sukhi:2:14',
    'sanjayo-na-sukhi:2:16', 'sanjayo-na-sukhi:3:5',
}


def build(book_dir):
    questions = {r['id']: r for r in book_questions(book_dir)}
    if not SELECTED <= questions.keys():
        raise ValueError('selected question missing')
    result = []
    for key in sorted(SELECTED):
        row = questions[key]
        prefix, number = key.rsplit(':', 1)
        target_id = prefix + ':' + str(int(number) + 1)
        answer = next((r for r in row['context'] if r['id'] == target_id), None)
        if answer is None or '?' in answer['iast']:
            raise ValueError(f'missing adjacent statement for {key}')
        result.append({'question_id': key, 'answer_id': target_id,
                       'question_iast': row['question_iast'],
                       'question_candidate': row['question_candidate'],
                       'answer_iast': answer['iast'],
                       'answer_candidate': answer['devanagari_candidate'],
                       'context': row['context'], 'source_url': row['source_url'],
                       'source_html_sha256': row['source_html_sha256'],
                       'category': 'story_comprehension',
                       'review_status': 'agent_selected_expert_review_pending',
                       'usage': 'draft_only_not_training_or_evaluation',
                       'caution': 'Question depends on story context; not a standalone fact. Transliteration and source terms need verification.'})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--book-dir', type=Path, default=ROOT / 'data/book_candidates/amarahasa_original')
    parser.add_argument('--out', type=Path, default=ROOT / 'data/qa_25k/story_seed_review.jsonl')
    args = parser.parse_args()
    rows = build(args.book_dir)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows), encoding='utf-8')
    print(f'{len(rows)} source-grounded draft pairs; expert review pending')


if __name__ == '__main__':
    main()
