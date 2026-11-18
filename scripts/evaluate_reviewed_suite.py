#!/usr/bin/env python3
"""Generate blinded-review inputs for a reviewed held-out evaluation suite.

This deliberately does not score answers or certify that a suite is held out.
A reviewer must check its provenance and mark the required attestations. Run on
both exported bundles with the same suite and compare using blinded reviewers.
"""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sys

CATEGORIES = {'conversation', 'knowledge', 'grammar', 'multiturn', 'reasoning'}


def load_suite(path):
    raw = path.read_bytes()
    rows = [json.loads(line) for line in raw.decode('utf-8').splitlines() if line.strip()]
    seen = set()
    counts = Counter()
    for row in rows:
        ident = row.get('id')
        if not isinstance(ident, str) or not ident or ident in seen:
            raise ValueError('each evaluation item needs a unique nonempty id')
        seen.add(ident)
        category = row.get('category')
        if category not in CATEGORIES:
            raise ValueError(f'{ident}: invalid category')
        if not isinstance(row.get('prompt'), str) or not row['prompt'].strip():
            raise ValueError(f'{ident}: missing prompt')
        if not isinstance(row.get('reference'), str) or not row['reference'].strip():
            raise ValueError(f'{ident}: missing reviewed reference')
        if row.get('reviewed') is not True or row.get('heldout_audited') is not True:
            raise ValueError(f'{ident}: review and held-out audit attestations required')
        if not isinstance(row.get('audit_evidence'), str) or not row['audit_evidence'].strip():
            raise ValueError(f'{ident}: identify audit evidence')
        history = row.get('history', [])
        if not isinstance(history, list) or any(not isinstance(t, dict) or not t.get('user') or not t.get('assistant') for t in history):
            raise ValueError(f'{ident}: invalid history')
        if category == 'multiturn' and not history:
            raise ValueError(f'{ident}: multi-turn item must include history')
        counts[category] += 1
    missing = {category: counts[category] for category in sorted(CATEGORIES) if counts[category] < 50}
    if missing:
        raise ValueError(f'at least 50 items in each category required: {missing}')
    return rows, hashlib.sha256(raw).hexdigest(), dict(counts)


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', type=Path, required=True)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    rows, suite_hash, counts = load_suite(args.suite)
    bundle = args.bundle.resolve()
    fingerprint = {name: sha256(bundle / name) for name in ('model.npz', 'tokenizer.json', 'meta.json')}
    os.environ['SERVE_DIR'] = str(bundle)
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    from api.index import do_chat
    results = []
    for item in rows:
        events = do_chat({'message': item['prompt'], 'history': item.get('history', []),
                          'temperature': 0.0, 'top_k': 40, 'top_p': 0.95,
                          'repetition_penalty': 1.15, 'max_new_tokens': 90, 'max_lines': 2})
        errors = [event.get('message', '') for event in events if event.get('event') == 'error']
        if errors:
            raise RuntimeError(f"{item['id']}: {errors}")
        done = [event for event in events if event.get('event') == 'done']
        if len(done) != 1:
            raise RuntimeError(f"{item['id']}: generation did not finish")
        results.append({'id': item['id'], 'category': item['category'], 'prompt': item['prompt'],
                        'history': item.get('history', []), 'reference': item['reference'],
                        'response': done[0]['text'], 'review_scores': None})
    output = {'suite_sha256': suite_hash, 'suite_counts': counts, 'bundle_sha256': fingerprint,
              'decoding': {'temperature': 0.0, 'top_k': 40, 'top_p': 0.95,
                           'repetition_penalty': 1.15, 'max_new_tokens': 90, 'max_lines': 2},
              'warning': 'Outputs are unscored. Attestations and leakage evidence require independent verification.',
              'results': results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temp = args.output.with_name(args.output.name + '.tmp')
    temp.write_text(json.dumps(output, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temp.replace(args.output)
    print(f'Wrote {len(results)} unscored responses to {args.output}')


if __name__ == '__main__':
    main()
