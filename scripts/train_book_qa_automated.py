#!/usr/bin/env python3
"""Experimental GitHub training on isolated source-grounded QA and book text.

Book text and QA are NOT independently reviewed. No evaluation or deployment.
Checkpoints include optimizer and random states; each saved segment is uploaded
with a checksum. A cancelled runner can lose steps since its last upload.
"""
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from sanskrit_llm.data import token_feature_table
from sanskrit_llm.model import ModelConfig, SanskritLM
from sanskrit_llm.tokenizer import AksharaTokenizer
from scripts.train_authored_qa import load_pairs

BOOK_DIR = ROOT / 'data/book_candidates/amarahasa_original'
CATALOGUE = ROOT / 'data/qa_25k/sktqa_pairs_review.jsonl'


def collect_examples(tok):
    examples = []
    counts = {'authored_qa': 0, 'published_qa': 0, 'book_chunks': 0, 'skipped': 0}
    seen = set()
    def add_qa(question, answer, source):
        q = ' '.join(question.split())
        if not q or not answer.strip() or q in seen or any(c in q + answer for c in '<>\n\r'):
            counts['skipped'] += 1
            return
        prefix = tok.encode(f'<user> {q} <assistant>')
        ans = tok.encode(' ' + answer.strip()) + [tok.eos_id]
        ids = prefix + ans
        if len(ids) - 1 > 256 or tok.unk_id in ids:
            counts['skipped'] += 1
            return
        seen.add(q)
        examples.append({'kind': 'qa', 'ids': ids, 'loss_start': len(prefix) - 1, 'source': source})
        counts[source] += 1
    for path in sorted((ROOT / 'data/qa_10k_seed').glob('authored_v*.tsv')):
        for row in load_pairs(path):
            add_qa(row['question'], row['answer'], 'authored_qa')
    # Retain only unambiguous single-answer, no-choice Ramayana records.
    # This remains a benchmark and is not an independent evaluation set.
    for line in CATALOGUE.read_text(encoding='utf-8').splitlines():
        row = json.loads(line)
        answer = row['answer_alternatives_raw']
        if ('sktqa/sanskrit.tsv:' not in row['id'] or row['choices_raw'] or ';' in answer or
                re.search(r'[A-Za-z]', row['question'] + answer)):
            continue
        add_qa(row['question'], answer, 'published_qa')
    book_manifest = json.loads((BOOK_DIR / 'manifest.json').read_text(encoding='utf-8'))
    for book in book_manifest['books']:
        path = BOOK_DIR / (book['book'] + '.jsonl')
        if hashlib.sha256(path.read_bytes()).hexdigest() != book['output_sha256']:
            raise ValueError(f'book checksum mismatch: {path}')
        chapters = {}
        for line in path.read_text(encoding='utf-8').splitlines():
            row = json.loads(line)
            chapter = row['id'].rsplit(':', 1)[0]
            chapters.setdefault(chapter, []).append(row['devanagari_candidate'].strip().replace('.', '।'))
        for lines in chapters.values():
            for i in range(0, len(lines), 3):
                ids = tok.encode(' '.join(lines[i:i+3]))
                # Isolated chunks, no attention between distinct examples.
                if not ids or tok.unk_id in ids:
                    counts['skipped'] += 1
                    continue
                for start in range(0, len(ids), 255):
                    chunk = [tok.bos_id] + ids[start:start+255] + [tok.eos_id]
                    # x contains at most 256 positions.
                    examples.append({'kind': 'book', 'ids': chunk, 'loss_start': 0, 'source': 'book_chunks'})
                    counts['book_chunks'] += 1
    if not all(counts[k] for k in ('authored_qa', 'published_qa', 'book_chunks')):
        raise ValueError(f'one source missing: {counts}')
    return examples, counts


def batch(examples, feat, pad_id, rng, batch_size):
    qa = [x for x in examples if x['kind'] == 'qa']
    books = [x for x in examples if x['kind'] == 'book']
    chosen = [rng.choice(books if rng.random() < 0.25 else qa) for _ in range(batch_size)]
    width = max(len(row['ids'])-1 for row in chosen)
    x = np.full((batch_size, width), pad_id, dtype=np.int64)
    y = np.full((batch_size, width), -1, dtype=np.int64)
    positions = loss_positions = 0
    counts = {'qa': 0, 'book': 0}
    for n, row in enumerate(chosen):
        ids = row['ids']; length = len(ids)-1; start = row['loss_start']
        x[n,:length] = ids[:-1]
        y[n,start:length] = ids[start+1:]
        positions += length
        loss_positions += length-start
        counts[row['kind']] += 1
    xt = torch.from_numpy(x)
    return xt, torch.from_numpy(y), feat[xt], positions, loss_positions, counts


def publish(path, release, name):
    import shutil
    asset = path.parent / name
    shutil.copyfile(path, asset)
    digest = hashlib.sha256(asset.read_bytes()).hexdigest()
    checksum = asset.with_name(asset.name + '.sha256')
    checksum.write_text(f'{digest}  {asset.name}\n', encoding='utf-8')
    subprocess.run(['gh','release','upload',release,str(asset),str(checksum)], check=True, cwd=ROOT)
    asset.unlink(); checksum.unlink()


def main():
    release = os.environ['TRAIN_RELEASE']
    run_id = os.environ['GITHUB_RUN_ID']
    base = ROOT / os.environ['BASE_ASSET']
    resume = os.environ.get('RESUME_ASSET', '')
    input_path = ROOT / resume if resume else base
    steps = int(os.environ.get('TRAIN_STEPS', '2000'))
    every = int(os.environ.get('CHECKPOINT_EVERY', '250'))
    tok = AksharaTokenizer.load(ROOT / 'artifacts/tokenizer.json')
    feat = token_feature_table(tok)
    examples, source_counts = collect_examples(tok)
    print('source counts:', json.dumps(source_counts), flush=True)
    fingerprint = hashlib.sha256(json.dumps(examples, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    torch.set_num_threads(2)
    torch.manual_seed(20261118)
    rng = random.Random(20261118)
    saved = torch.load(input_path, map_location='cpu', weights_only=False)
    if not saved.get('optimizer'):
        raise ValueError('checkpoint lacks optimizer')
    if resume:
        if saved.get('metadata', {}).get('dataset_sha256') != fingerprint:
            raise ValueError('resume dataset differs from original checkpoint')
    elif int(saved.get('step', -1)) != 40:
        raise ValueError('expected 40-step authored QA checkpoint')
    model = SanskritLM(ModelConfig(**saved['config']))
    model.load_state_dict(saved['state_dict'], strict=True)
    decay, no_decay = [], []
    for _, p in model.named_parameters():
        (decay if p.dim() >= 2 else no_decay).append(p)
    opt = torch.optim.AdamW([{'params': decay, 'weight_decay': 0.1},
                             {'params': no_decay, 'weight_decay': 0.0}], lr=1e-5, betas=(0.9, 0.95), eps=1e-8)
    opt.load_state_dict(saved['optimizer'])
    for group in opt.param_groups:
        group['lr'] = 1e-5
    out = ROOT / 'artifacts/automated_book_qa'
    out.mkdir(parents=True, exist_ok=True)
    positions = loss_positions = 0
    sampled = {'qa': 0, 'book': 0}
    start_step = 0
    if resume:
        meta = saved['metadata']
        start_step = int(saved['step'])
        positions = int(meta['nonpadding_positions'])
        loss_positions = int(meta['assistant_and_book_loss_positions'])
        sampled = dict(meta['sampled_examples'])
        rng.setstate(saved['python_random_state'])
        torch.set_rng_state(saved['torch_rng_state'])
    if start_step >= steps:
        raise ValueError('checkpoint already at or beyond requested target')
    model.train()
    for step in range(start_step+1, steps+1):
        x, y, f, n, nl, types = batch(examples, feat, tok.pad_id, rng, 8)
        _, loss, _ = model(x, f, targets=y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        positions += n; loss_positions += nl
        for kind in sampled:
            sampled[kind] += types[kind]
        if step == 1 or step % 50 == 0:
            print(f'step {step}/{steps}, loss={float(loss):.4f}, nonpad_positions={positions}', flush=True)
        if step % every == 0 or step == steps:
            meta = {'step': step, 'base_asset': base.name, 'base_approx_positions': 220579586,
                    'nonpadding_positions': positions, 'assistant_and_book_loss_positions': loss_positions,
                    'approx_total_positions': 220579586 + positions, 'sampled_examples': sampled.copy(),
                    'source_counts': source_counts, 'dataset_sha256': fingerprint,
                    'qa_probability': 0.75, 'book_probability': 0.25,
                    'heldout_evaluation_complete': False, 'deployed': False,
                    'warning': 'Unreviewed texts; sampled repeatedly. No independent quality or grammar retention evaluation.'}
            tmp = out / 'latest.tmp'; latest = out / 'latest.pt'
            torch.save({'config': model.cfg.to_dict(), 'state_dict': model.state_dict(),
                        'optimizer': opt.state_dict(), 'step': step, 'metadata': meta,
                        'python_random_state': rng.getstate(), 'torch_rng_state': torch.get_rng_state()}, tmp)
            os.replace(tmp, latest)
            name = f'book-qa-auto-{run_id}-step-{step:05d}.pt'
            publish(latest, release, name)
            summary = out / f'book-qa-auto-{run_id}-step-{step:05d}.json'
            summary.write_text(json.dumps(meta, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
            subprocess.run(['gh','release','upload',release,str(summary)], check=True, cwd=ROOT)
            print('published checkpoint:', json.dumps(meta, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
