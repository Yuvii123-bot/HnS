#!/usr/bin/env python3
"""Audit authored QA batches, without declaring them accurate or held out."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys
import unicodedata

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.train_authored_qa import load_pairs


def norm(text):
    return ' '.join(unicodedata.normalize('NFC', text).split())


def audit(paths, eval_paths):
    seen = defaultdict(list)
    categories = Counter()
    manifest = []
    for path in paths:
        rows = load_pairs(path)
        manifest.append({'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'rows': len(rows)})
        for row in rows:
            seen[norm(row['question'])].append((str(path), row['answer'], row['category']))
            categories[row['category']] += 1
    duplicates = {q: rows for q, rows in seen.items() if len(rows) > 1}
    conflict = {q: rows for q, rows in duplicates.items() if len({norm(x[1]) for x in rows}) > 1}
    overlap = []
    for path in eval_paths:
        for line in path.read_text(encoding='utf-8').splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            prompt = row.get('prompt')
            if isinstance(prompt, str) and norm(prompt) in seen:
                overlap.append({'file': str(path), 'id': row.get('id'), 'prompt_sha256': hashlib.sha256(norm(prompt).encode()).hexdigest()})
    return {'total_rows': sum(categories.values()), 'categories': dict(categories),
            'unique_questions': len(seen), 'duplicate_question_groups': len(duplicates),
            'conflicting_answer_groups': len(conflict), 'exact_eval_prompt_matches': overlap,
            'sources': manifest, 'target': 10000, 'target_grammar': 100,
            'warning': 'Exact matches only. Source quality, semantic duplicates, correctness and historical training overlap remain unverified; old evaluation sets may themselves be contaminated.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch', action='append', type=Path, required=True)
    parser.add_argument('--eval', action='append', type=Path, default=[])
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    report = audit(args.batch, args.eval)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
