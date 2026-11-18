#!/usr/bin/env python3
"""Find exact prompt substrings in corpus text; not a semantic leakage test.

Requires pyahocorasick. Reports counts and item IDs, not model accuracy. Absence
of an exact match never proves an item is held out.
"""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

import ahocorasick


def check(files, parts):
    prompts = defaultdict(list)
    totals = Counter()
    for path in files:
        with path.open(encoding='utf-8') as fh:
            for line in fh:
                if not line.strip():
                    continue
                item = json.loads(line)
                prompt = item.get('prompt', '').strip()
                if prompt:
                    key = (str(path), str(item['id']))
                    prompts[prompt].append(key)
                    totals[str(path)] += 1
    automaton = ahocorasick.Automaton()
    for prompt, ids in prompts.items():
        automaton.add_word(prompt, ids)
    automaton.make_automaton()
    hits = set()
    for path in parts:
        with path.open(encoding='utf-8') as fh:
            for line in fh:
                for _, ids in automaton.iter(line):
                    hits.update(ids)
    return {'input_parts': len(parts), 'totals': dict(totals),
            'matching_counts': {str(path): sum(source == str(path) for source, _ in hits) for path in files},
            'matched_ids': {str(path): sorted(ident for source, ident in hits if source == str(path)) for path in files},
            'limitation': 'Exact substring only; no match does not establish held-out status.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--eval', type=Path, required=True, action='append')
    parser.add_argument('--corpus', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    parts = sorted(args.corpus.glob('part_*.txt'))
    if not parts:
        parser.error('no corpus parts found')
    out = json.dumps(check(args.eval, parts), ensure_ascii=False, indent=2) + '\n'
    if args.output:
        args.output.write_text(out, encoding='utf-8')
    else:
        print(out, end='')


if __name__ == '__main__':
    main()
