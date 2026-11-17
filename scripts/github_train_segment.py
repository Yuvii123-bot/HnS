#!/usr/bin/env python3
"""Run bounded training segments and publish durable checkpoints to a release.

Each published checkpoint includes model weights and the optimizer. A segment
interrupted before upload can lose work since its last published checkpoint.
"""
import hashlib
import os
import subprocess
import sys
from pathlib import Path

import torch

stage = int(os.environ['STAGE'])
release = os.environ['TRAIN_RELEASE']
root = Path(__file__).resolve().parents[1]
out = root / 'artifacts/run250'
out.mkdir(parents=True, exist_ok=True)
ckpt = out / 'latest.pt'


def run(*args):
    subprocess.run(args, check=True, cwd=root)


def step():
    blob = torch.load(ckpt, map_location='cpu', weights_only=False)
    return int(blob['step'])


if stage > 1:
    asset = f'checkpoint-{stage-1:02d}-final.pt'
    run('gh', 'release', 'download', release, '--pattern', asset, '--pattern', asset + '.sha256', '--dir', str(out))
    expected = (out / (asset + '.sha256')).read_text().split()[0]
    with (out / asset).open('rb') as fh:
        actual = hashlib.file_digest(fh, 'sha256').hexdigest()
    if actual != expected:
        raise RuntimeError('Downloaded checkpoint checksum mismatch')
    (out / asset).rename(ckpt)
else:
    import numpy as np
    import json
    sys.path.insert(0, str(root))
    from sanskrit_llm.model import ModelConfig, SanskritLM
    meta = json.loads((root / 'serve/meta.json').read_text())
    model = SanskritLM(ModelConfig(**meta['config']))
    with np.load(root / 'serve/model.npz', allow_pickle=False) as source:
        weights = {key: torch.from_numpy(source[key].copy()) for key in source.files}
    model.load_state_dict(weights, strict=True)
    del weights, model
    # Reload directly; the initial exported weights have no optimizer state.
    with np.load(root / 'serve/model.npz', allow_pickle=False) as source:
        weights = {key: torch.from_numpy(source[key].copy()) for key in source.files}
    torch.save({'config': meta['config'], 'state_dict': weights, 'optimizer': None,
                'step': int(meta['train_summary']['steps'])}, ckpt)

for segment in range(1, 5):
    before = step()
    if before >= 61036:
        break
    cmd = [sys.executable, 'scripts/train_model.py', '--bin-dir', 'data/bin_250m',
           '--out', str(out), '--steps', '61036', '--batch', '16', '--ctx', '256',
           '--d-model', '288', '--layers', '8', '--heads', '8', '--d-ff', '880',
           '--lr', '0.001', '--warmup', '2000', '--threads', '2',
           '--eval-every', '4000', '--ckpt-every', '200', '--resume', str(ckpt)]
    try:
        result = subprocess.run(cmd, cwd=root, timeout=3600)
        if result.returncode:
            raise RuntimeError(f'trainer failed: {result.returncode}')
    except subprocess.TimeoutExpired:
        print('Time segment complete; using last atomic checkpoint.', flush=True)
    after = step()
    if after <= before:
        raise RuntimeError(f'No checkpoint progress: {before} -> {after}')
    asset = out / f'checkpoint-{stage:02d}-{segment}.pt'
    # Copy first so a later segment never mutates the uploaded file.
    import shutil
    shutil.copyfile(ckpt, asset)
    digest = hashlib.file_digest(asset.open('rb'), 'sha256').hexdigest()
    checksum = asset.with_suffix(asset.suffix + '.sha256')
    checksum.write_text(f'{digest}  {asset.name}\n')
    run('gh', 'release', 'upload', release, str(asset), str(checksum))
    print(f'Published step {after}: {after*4096:,} positions', flush=True)
    asset.unlink()
    checksum.unlink()

final = out / f'checkpoint-{stage:02d}-final.pt'
import shutil
shutil.copyfile(ckpt, final)
with final.open('rb') as fh:
    digest = hashlib.file_digest(fh, 'sha256').hexdigest()
checksum = final.with_suffix(final.suffix + '.sha256')
checksum.write_text(f'{digest}  {final.name}\n')
run('gh', 'release', 'upload', release, str(final), str(checksum))
print(f'Stage {stage} completed at checkpoint step {step()}', flush=True)
