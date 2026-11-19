#!/usr/bin/env python3
"""Experimental CPU fine-tuning with a sourced, unreviewed 100-question batch.

Retains optimizer state; isolates old and new checkpoints. QA loss masks prompts.
The new questions and the old replay examples are not independent evaluation data.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from sanskrit_llm.data import token_feature_table
from sanskrit_llm.model import ModelConfig, SanskritLM
from sanskrit_llm.tokenizer import AksharaTokenizer
from scripts.train_book_qa_automated import collect_examples

DATA = ROOT / 'data/ramayana_qa_authored/balakanda_intro_v1.jsonl'


def read_batch(tok, start, count):
    rows = [json.loads(line) for line in DATA.read_text(encoding='utf-8').splitlines() if line]
    rows = rows[start:start+count]
    if len(rows) != count:
        raise ValueError(f'Expected {count} authored rows at offset {start}; only {len(rows)} available')
    examples = []
    for row in rows:
        if row['status'] != 'authored_unreviewed' or row['split'] != 'quarantine':
            raise ValueError('Unexpected source status')
        prefix = tok.encode('<user> ' + row['question'].strip() + ' <assistant>')
        answer = tok.encode(' ' + row['answer'].strip()) + [tok.eos_id]
        ids = prefix + answer
        if tok.unk_id in ids or len(ids)-1 > 256 or not answer[:-1]:
            raise ValueError('Unusable question: ' + row['id'])
        examples.append({'ids': ids, 'loss_start': len(prefix)-1, 'kind': 'qa', 'source': 'ramayana_qa'})
    return examples, rows


def make_batch(new, old, feat, pad_id, rng, size):
    # Each example is isolated; half is new QA, half is replay, randomly selected.
    selected = [rng.choice(new if rng.random() < 0.5 else old) for _ in range(size)]
    width = max(len(row['ids'])-1 for row in selected)
    x = np.full((size, width), pad_id, dtype=np.int64)
    y = np.full((size, width), -1, dtype=np.int64)
    n = loss_n = new_n = 0
    for i, row in enumerate(selected):
        ids, start = row['ids'], row['loss_start']
        length = len(ids)-1
        x[i, :length] = ids[:-1]
        y[i, start:length] = ids[start+1:]
        n += length; loss_n += length-start
        new_n += row['source'] == 'ramayana_qa'
    x = torch.from_numpy(x)
    return x, torch.from_numpy(y), feat[x], n, loss_n, new_n


def run():
    p = argparse.ArgumentParser()
    p.add_argument('--base', type=Path, required=True)
    p.add_argument('--offset', type=int, default=0)
    p.add_argument('--count', type=int, default=100)
    p.add_argument('--steps', type=int, default=100)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--save-every', type=int, default=50)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--release', required=True)
    a = p.parse_args()
    if a.offset < 0 or min(a.count, a.steps, a.batch_size, a.save_every) < 1:
        p.error('Invalid training arguments')
    run_id = os.environ.get('GITHUB_RUN_ID', '')
    if not run_id.isdigit():
        raise RuntimeError('Requires a GitHub Actions run ID to name checkpoints')
    torch.set_num_threads(2)
    tok = AksharaTokenizer.load(ROOT / 'artifacts/tokenizer.json')
    features = token_feature_table(tok)
    new, rows = read_batch(tok, a.offset, a.count)
    all_old, old_counts = collect_examples(tok)
    # Replay prior authored QA and book data; omit the benchmark-derived SktQA.
    old = [row for row in all_old if row['source'] in ('authored_qa', 'book_chunks')]
    if not old:
        raise ValueError('No replay data available')
    fingerprint = hashlib.sha256(json.dumps(rows, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    saved = torch.load(a.base, map_location='cpu', weights_only=False)
    meta = saved['metadata']
    if meta.get('kind') not in ('ramayana_v1', 'ramayana_qa_v1') or not saved.get('optimizer'):
        raise ValueError('Expected a complete Ramayana checkpoint including optimizer')
    base_total = int(meta['approx_total_positions'])
    if base_total < 315275224:
        raise ValueError('Base predates deployed Ramayana model')
    model = SanskritLM(ModelConfig(**saved['config']))
    model.load_state_dict(saved['state_dict'], strict=True)
    decay, no_decay = [], []
    for param in model.parameters():
        (decay if param.dim() >= 2 else no_decay).append(param)
    opt = torch.optim.AdamW([{'params': decay, 'weight_decay': 0.1}, {'params': no_decay, 'weight_decay': 0.0}],
                            lr=1e-5, betas=(0.9, 0.95), eps=1e-8)
    opt.load_state_dict(saved['optimizer'])
    for group in opt.param_groups:
        group['lr'] = 1e-5
    rng = random.Random(); rng.setstate(saved['python_random_state'])
    torch.set_rng_state(saved['torch_rng_state'])
    positions = loss_positions = new_samples = 0
    a.out.mkdir(parents=True, exist_ok=True)
    model.train()
    print(f'Base positions: {base_total:,}; new authored QA: {len(new)}; replay examples: {len(old)}; fingerprint: {fingerprint}', flush=True)
    for step in range(1, a.steps+1):
        x, y, f, n, nl, nn = make_batch(new, old, features, tok.pad_id, rng, a.batch_size)
        opt.zero_grad(set_to_none=True)
        _, loss, _ = model(x, f, targets=y)
        if not torch.isfinite(loss):
            raise RuntimeError('Non-finite loss; refusing to publish invalid weights')
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not torch.isfinite(grad_norm):
            raise RuntimeError('Non-finite gradient; refusing to publish invalid weights')
        opt.step(); positions += n; loss_positions += nl; new_samples += nn
        if step == 1 or step % 10 == 0:
            print(f'step={step}/{a.steps} loss={float(loss):.5f} input_positions={positions:,} loss_positions={loss_positions:,} new_qa_samples={new_samples}', flush=True)
        if step % a.save_every == 0 or step == a.steps:
            result = {'kind': 'ramayana_qa_v1', 'base_checkpoint': a.base.name,
                      'base_checkpoint_sha256': hashlib.sha256(a.base.read_bytes()).hexdigest(),
                      'base_total_positions': base_total, 'additional_nonpadding_positions': positions,
                      'additional_loss_positions': loss_positions, 'approx_total_positions': base_total + positions,
                      'round_offset': a.offset, 'round_count': a.count, 'round_steps': step,
                      'new_qa_samples': new_samples, 'dataset_sha256': fingerprint,
                      'batch_ids': [row['id'] for row in rows], 'old_source_counts': old_counts,
                      'replay_source_filter': ['authored_qa', 'book_chunks'],
                      'heldout_evaluation_complete': False,
                      'warning': 'Assistant-authored QA unreviewed; repeated sampling; CPU continuation; no held-out or grammar-retention evaluation.'}
            name = f'ramayana-qa-{run_id}-step-{step:05d}.pt'
            dest = a.out/name
            temporary = a.out/(name+'.tmp')
            torch.save({'config': model.cfg.to_dict(), 'state_dict': model.state_dict(), 'optimizer': opt.state_dict(),
                        'metadata': result, 'python_random_state': rng.getstate(), 'torch_rng_state': torch.get_rng_state(),
                        'cuda_rng_state': saved.get('cuda_rng_state'), 'scaler': saved.get('scaler')}, temporary)
            os.replace(temporary, dest)
            digest = hashlib.sha256(dest.read_bytes()).hexdigest()
            sidecar = a.out/(name+'.sha256'); sidecar.write_text(f'{digest}  {name}\n')
            summary = a.out/(name+'.json'); summary.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
            subprocess.run(['gh', 'release', 'upload', a.release, str(dest), str(sidecar), str(summary)], check=True, cwd=ROOT)
            print('Published verified candidate:', name, 'sha256:', digest, 'total positions:', base_total+positions, flush=True)

if __name__ == '__main__':
    run()
