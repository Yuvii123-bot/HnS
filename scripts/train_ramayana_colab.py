#!/usr/bin/env python3
"""Separate, resumable Ramayana fine-tuning from a checksummed Colab checkpoint."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from sanskrit_llm.data import token_feature_table
from sanskrit_llm.model import ModelConfig, SanskritLM
from sanskrit_llm.tokenizer import AksharaTokenizer
from scripts.colab_train_1b import file_hash, latest_checkpoint, save_checkpoint, verify_checkpoint

SOURCE_REV = 'd8a6ccec70099d8f2e23aa3d7fdf439d336b4116'


def prepare(source, tok):
    examples, skipped, seen = [], 0, set()
    paths = sorted(source.glob('*kanda_sanskrit.txt'))
    if len(paths) != 5:
        raise RuntimeError('Expected the five pinned source books; check corpus checkout')
    raw_hashes = {p.name: file_hash(p) for p in paths}
    for path in paths:
        for line in path.read_text(encoding='utf-8').splitlines():
            text = ' '.join(line.split())
            if not text or text in seen:
                continue
            seen.add(text)
            ids = tok.encode(text)
            if not ids or tok.unk_id in ids:
                skipped += 1
                continue
            for start in range(0, len(ids), 255):
                examples.append([tok.bos_id] + ids[start:start + 255] + [tok.eos_id])
    if not examples:
        raise RuntimeError('No usable Ramayana text')
    digest = hashlib.sha256(json.dumps(examples, separators=(',', ':')).encode()).hexdigest()
    return examples, {'books': len(paths), 'chunks': len(examples), 'unique_lines': len(seen),
                      'unknown_token_lines_skipped': skipped, 'source_files_sha256': raw_hashes,
                      'dataset_sha256': digest, 'source_revision': SOURCE_REV}


def make_batch(examples, features, pad, rng, size):
    chosen = [rng.choice(examples) for _ in range(size)]
    width = max(len(x) - 1 for x in chosen)
    x = np.full((size, width), pad, dtype=np.int64)
    y = np.full((size, width), -1, dtype=np.int64)
    positions = 0
    for i, ids in enumerate(chosen):
        n = len(ids) - 1
        x[i, :n], y[i, :n] = ids[:-1], ids[1:]
        positions += n
    x = torch.from_numpy(x)
    return x, torch.from_numpy(y), features[x], positions


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--base-dir', type=Path, required=True)
    p.add_argument('--out-dir', type=Path, required=True)
    p.add_argument('--target', type=int, default=10_000_000)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--save-seconds', type=int, default=300)
    p.add_argument('--lr', type=float, default=1e-5)
    a = p.parse_args()
    if min(a.target, a.batch_size, a.save_seconds) <= 0 or a.lr <= 0:
        p.error('All training settings must be positive')
    if a.out_dir.resolve() == a.base_dir.resolve():
        raise RuntimeError('Output must be a separate folder from the original training run')
    if not torch.cuda.is_available():
        raise RuntimeError('GPU required; choose a GPU runtime in Colab')
    tok = AksharaTokenizer.load(ROOT / 'artifacts/tokenizer.json')
    examples, data = prepare(a.source, tok)
    print('Corpus:', json.dumps(data, ensure_ascii=False), flush=True)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    previous = latest_checkpoint(a.out_dir)
    base = previous or latest_checkpoint(a.base_dir)
    if base is None:
        raise RuntimeError('No full Colab checkpoint found in the original folder')
    # Snapshot the base so a concurrently saving trainer cannot affect our load.
    verify_checkpoint(base)
    expected = Path(str(base) + '.sha256').read_text().split()[0]
    local = Path('/content/ramayana-start.pt')
    if not local.parent.exists():
        local = Path('/tmp/ramayana-start.pt')
    shutil.copyfile(base, local)
    if file_hash(local) != expected:
        raise RuntimeError('Copied checkpoint failed checksum; retry')
    saved = torch.load(local, map_location='cpu', weights_only=False)
    local.unlink()
    old = saved.get('metadata', {})
    if not saved.get('optimizer'):
        raise RuntimeError('Full optimizer checkpoint required')
    if previous:
        if old.get('kind') != 'ramayana_v1' or old.get('dataset_sha256') != data['dataset_sha256']:
            raise RuntimeError('Resume checkpoint dataset or format mismatch')
        if old['batch_size'] != a.batch_size or old['learning_rate'] != a.lr:
            raise RuntimeError('Resume settings differ from checkpoint')
        base_total = int(old['base_total_positions'])
        total, step = int(old['additional_nonpadding_positions']), int(old['ramayana_steps'])
        base_name, base_sha = old['base_checkpoint'], old['base_checkpoint_sha256']
    else:
        if old.get('kind') != 'colab_1b_v1':
            raise RuntimeError('Expected a full checkpoint from the original Colab run')
        base_total = int(old['approx_total_positions'])
        total, step, base_name, base_sha = 0, 0, base.name, expected
    model = SanskritLM(ModelConfig(**saved['config'])).cuda()
    model.load_state_dict(saved['state_dict'], strict=True)
    decay, no_decay = [], []
    for param in model.parameters():
        (decay if param.dim() >= 2 else no_decay).append(param)
    opt = torch.optim.AdamW([{'params': decay, 'weight_decay': 0.1}, {'params': no_decay, 'weight_decay': 0.0}], lr=a.lr, betas=(0.9, 0.95), eps=1e-8)
    opt.load_state_dict(saved['optimizer'])
    for group in opt.param_groups:
        group['lr'] = a.lr
    rng = random.Random()
    rng.setstate(saved['python_random_state'])
    torch.set_rng_state(saved['torch_rng_state'])
    torch.cuda.set_rng_state_all(saved['cuda_rng_state'])
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.cuda.amp.GradScaler(enabled=dtype == torch.float16)
    if saved.get('scaler') and scaler.is_enabled():
        scaler.load_state_dict(saved['scaler'])
    features = token_feature_table(tok)
    print(f'Base: {base_name} ({base_sha}); base positions: {base_total:,}; Ramayana positions: {total:,}/{a.target:,}', flush=True)
    model.train()
    last_save, last_time = total, time.monotonic()

    def save():
        meta = {'kind': 'ramayana_v1', 'base_checkpoint': base_name, 'base_checkpoint_sha256': base_sha,
                'base_total_positions': base_total, 'additional_nonpadding_positions': total,
                'additional_loss_positions': total, 'approx_total_positions': base_total + total,
                'ramayana_steps': step, 'target_additional_positions': a.target,
                'batch_size': a.batch_size, 'learning_rate': a.lr,
                'heldout_evaluation_complete': False, **data,
                'warning': 'Repeated unreviewed verses; grammar retention and held-out accuracy untested.'}
        result = save_checkpoint(a.out_dir, {'config': model.cfg.to_dict(), 'state_dict': model.state_dict(),
            'optimizer': opt.state_dict(), 'metadata': meta, 'python_random_state': rng.getstate(),
            'torch_rng_state': torch.get_rng_state(), 'cuda_rng_state': torch.cuda.get_rng_state_all(),
            'scaler': scaler.state_dict()})
        print(f'Saved {result}; Ramayana positions={total:,}; approximate total={base_total+total:,}', flush=True)

    try:
        while total < a.target:
            x, y, f, n = make_batch(examples, features, tok.pad_id, rng, a.batch_size)
            x, y, f = x.cuda(), y.cuda(), f.cuda()
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type='cuda', dtype=dtype):
                _, loss, _ = model(x, f, targets=y)
            if not torch.isfinite(loss):
                raise RuntimeError('Non-finite loss; not saving invalid weights')
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scale_before = scaler.get_scale()
            scaler.step(opt)
            scaler.update()
            if scaler.is_enabled() and scaler.get_scale() < scale_before:
                continue
            total += n
            step += 1
            if step % 100 == 0:
                print(f'step={step} loss={float(loss):.4f} Ramayana positions={total:,}/{a.target:,} total={base_total+total:,}', flush=True)
            if time.monotonic() - last_time >= a.save_seconds:
                save()
                last_save, last_time = total, time.monotonic()
    except KeyboardInterrupt:
        if total != last_save:
            save()
        print('Interrupted; rerun to resume.', flush=True)
        return
    if total != last_save:
        save()
    print('Target reached.', flush=True)

if __name__ == '__main__':
    main()
