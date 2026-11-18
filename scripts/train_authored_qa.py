#!/usr/bin/env python3
"""Fine-tune on a small authored QA seed, one example per attention context.

Experimental only: no automatic evaluation or deployment. Prefix tokens are
masked from loss and padding is excluded; EOS is included in assistant loss.
"""
import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from sanskrit_llm.data import token_feature_table
from sanskrit_llm.model import ModelConfig, SanskritLM
from sanskrit_llm.tokenizer import AksharaTokenizer
from sanskrit_llm.trainer import train


def load_pairs(path):
    with path.open(encoding='utf-8', newline='') as source:
        reader = csv.DictReader(source, delimiter='\t')
        if reader.fieldnames != ['category', 'question', 'answer']:
            raise ValueError('unexpected TSV columns')
        rows = list(reader)
    seen = set()
    for row in rows:
        if None in row or not all(row[k].strip() for k in ('category', 'question', 'answer')):
            raise ValueError('empty or malformed row')
        question = ' '.join(row['question'].split())
        if question in seen:
            raise ValueError(f'duplicate question: {question}')
        seen.add(question)
        if any('<' in row[k] or '>' in row[k] for k in ('question', 'answer')):
            raise ValueError('template markup in source')
    return rows


class IsolatedPairs:
    def __init__(self, rows, tokenizer, feature_table, block_size=256, seed=1337):
        self.block_size, self.feat_table = block_size, feature_table
        self.pad_id = tokenizer.pad_id
        self.rng = np.random.default_rng(seed)
        self.examples = []
        self.positions = 0
        self.loss_positions = 0
        for row in rows:
            prefix = tokenizer.encode(f"<user> {row['question']} <assistant>")
            answer = tokenizer.encode(' ' + row['answer']) + [tokenizer.eos_id]
            ids = prefix + answer
            if len(ids) < 2 or len(ids) - 1 > block_size or not answer:
                raise ValueError(f'example too long or empty: {row["question"]}')
            self.examples.append((ids, len(prefix)))
        if not self.examples:
            raise ValueError('empty data')

    def sample_batch(self, batch_size):
        selected = [self.examples[int(i)] for i in self.rng.integers(0, len(self.examples), size=batch_size)]
        width = max(len(ids) - 1 for ids, _ in selected)
        x = np.full((batch_size, width), self.pad_id, dtype=np.int64)
        y = np.full((batch_size, width), -1, dtype=np.int64)
        for i, (ids, prefix_len) in enumerate(selected):
            length = len(ids) - 1
            x[i, :length] = ids[:-1]
            # Target at position prefix_len-1 is first answer token.
            y[i, prefix_len-1:length] = ids[prefix_len:]
            self.positions += length
            self.loss_positions += length - (prefix_len - 1)
        xt, yt = torch.from_numpy(x), torch.from_numpy(y)
        return xt, yt, self.feat_table[xt]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pairs', type=Path, required=True)
    parser.add_argument('--tokenizer', type=Path, default=Path('artifacts/tokenizer.json'))
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=40)
    parser.add_argument('--batch', type=int, default=8)
    args = parser.parse_args()
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    rows = load_pairs(args.pairs)
    tok = AksharaTokenizer.load(args.tokenizer)
    torch.manual_seed(1337)
    blob = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    if int(blob.get('step', -1)) != 100 or not blob.get('optimizer'):
        raise ValueError('unexpected book checkpoint or missing optimizer')
    cfg = ModelConfig(**blob['config'])
    dataset = IsolatedPairs(rows, tok, token_feature_table(tok), block_size=256)
    summary = train(SanskritLM(cfg), dataset, None, tok, steps=args.steps, batch_size=args.batch,
                    lr=1e-5, warmup=5, out_dir=args.out, resume=args.checkpoint,
                    reset_steps=True, threads=2, eval_every=0, ckpt_every=10)
    summary.update(authored_qa_rows=len(rows), grammar_rows=sum(r['category'] == 'grammar' for r in rows),
                   actual_nonpadding_input_positions=dataset.positions,
                   assistant_and_eos_loss_positions=dataset.loss_positions,
                   nominal_base_positions=220569600, heldout_evaluation_complete=False,
                   warning='Small authored seed, not independently reviewed. No evaluation or deployment.')
    (args.out / 'authored_qa_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
