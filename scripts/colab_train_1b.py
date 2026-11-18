#!/usr/bin/env python3
"""GPU fine-tuning from the saved book/QA candidate, checkpointed to Drive.

Token target is ADDITIONAL non-padding model-input positions, including the
QA prompt positions whose loss is masked. Loss-bearing tokens are counted
separately. Data is repeated sampling, not one billion distinct tokens.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from sanskrit_llm.data import token_feature_table
from sanskrit_llm.model import ModelConfig, SanskritLM
from sanskrit_llm.tokenizer import AksharaTokenizer
from scripts.train_book_qa_automated import batch, collect_examples


def file_hash(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def verify_checkpoint(path):
    sidecar = Path(str(path) + '.sha256')
    if not sidecar.is_file():
        raise RuntimeError(f'Checkpoint has no checksum: {path}')
    expected = sidecar.read_text(encoding='utf-8').strip().split()[0]
    if file_hash(path) != expected:
        raise RuntimeError(f'Checkpoint checksum mismatch: {path}')


def latest_checkpoint(directory):
    # Names are zero-padded processed token counts; ordering is stable.
    files = sorted(directory.glob('tokens-*.pt'))
    if not files:
        return None
    latest = files[-1]
    verify_checkpoint(latest)  # fail rather than silently reverting progress
    return latest


def save_checkpoint(directory, payload):
    directory.mkdir(parents=True, exist_ok=True)
    count = payload['metadata']['additional_nonpadding_positions']
    path = directory / f'tokens-{count:012d}.pt'
    if path.exists() or Path(str(path) + '.sha256').exists():
        raise FileExistsError(f'Refusing to overwrite a checkpoint: {path}')
    temporary = directory / (path.name + '.tmp')
    with temporary.open('wb') as fh:
        torch.save(payload, fh)
        fh.flush()
        os.fsync(fh.fileno())
    digest = file_hash(temporary)
    os.replace(temporary, path)
    sidecar = Path(str(path) + '.sha256')
    side_tmp = Path(str(sidecar) + '.tmp')
    with side_tmp.open('w', encoding='utf-8') as fh:
        fh.write(f'{digest}  {path.name}\n')
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(side_tmp, sidecar)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', required=True, type=Path, help='Verified GitHub checkpoint')
    parser.add_argument('--drive-dir', required=True, type=Path)
    parser.add_argument('--target-tokens', type=int, default=1_000_000_000)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--save-seconds', type=int, default=300)
    parser.add_argument('--lr', type=float, default=1e-5)
    args = parser.parse_args()
    if args.target_tokens <= 0 or args.batch_size <= 0 or args.save_seconds <= 0 or args.lr <= 0:
        parser.error('target, batch size, save interval and learning rate must be positive')
    if not torch.cuda.is_available():
        raise RuntimeError('No GPU detected. In Colab select Runtime > Change runtime type > GPU.')
    verify_checkpoint(args.base)
    device = torch.device('cuda')
    tok = AksharaTokenizer.load(ROOT / 'artifacts/tokenizer.json')
    feat = token_feature_table(tok)
    examples, counts = collect_examples(tok)
    fingerprint = hashlib.sha256(json.dumps(examples, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    previous = latest_checkpoint(args.drive_dir)
    input_path = previous or args.base
    saved = torch.load(input_path, map_location='cpu', weights_only=False)
    if not saved.get('optimizer'):
        raise RuntimeError('Checkpoint is missing optimizer state')
    old = saved.get('metadata', {})
    if old.get('dataset_sha256') != fingerprint:
        raise RuntimeError('Dataset differs from checkpoint. Refusing to resume on a different mix.')
    if previous is None and int(saved.get('step', -1)) != 2000:
        raise RuntimeError('Base is not the expected 2000-step GitHub candidate')
    if previous and old.get('kind') != 'colab_1b_v1':
        raise RuntimeError('Unexpected Drive checkpoint format')
    model = SanskritLM(ModelConfig(**saved['config'])).to(device)
    model.load_state_dict(saved['state_dict'], strict=True)
    decay, no_decay = [], []
    for _, p in model.named_parameters():
        (decay if p.dim() >= 2 else no_decay).append(p)
    opt = torch.optim.AdamW([{'params': decay, 'weight_decay': 0.1},
                             {'params': no_decay, 'weight_decay': 0.0}], lr=args.lr,
                            betas=(0.9, 0.95), eps=1e-8)
    opt.load_state_dict(saved['optimizer'])
    for group in opt.param_groups:
        group['lr'] = args.lr
    rng = random.Random()
    rng.setstate(saved['python_random_state'])
    torch.set_rng_state(saved['torch_rng_state'])
    if previous:
        if old['batch_size'] != args.batch_size or old['learning_rate'] != args.lr:
            raise RuntimeError('Batch size or learning rate differs from saved run; refusing silent change.')
        torch.cuda.set_rng_state_all(saved['cuda_rng_state'])
    else:
        torch.cuda.manual_seed_all(20261118)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    # The CUDA-specific API also works with older Colab PyTorch versions.
    scaler = torch.cuda.amp.GradScaler(enabled=(dtype == torch.float16))
    if previous and saved.get('scaler'):
        scaler.load_state_dict(saved['scaler'])
    base_positions = int(old.get('base_total_positions', old.get('approx_total_positions', 0)))
    if not base_positions:
        raise RuntimeError('Missing base token count')
    total = int(old.get('additional_nonpadding_positions', 0))
    loss_total = int(old.get('additional_loss_positions', 0))
    step = int(old.get('colab_steps', 0))
    last_saved = total
    last_save_time = time.monotonic()
    print(f'GPU: {torch.cuda.get_device_name(0)}; precision: {dtype}', flush=True)
    print(f'Loaded: {input_path}; base positions: {base_positions:,}; '
          f'additional: {total:,}/{args.target_tokens:,}; source counts: {counts}', flush=True)
    model.train()

    def save():
        meta = {'kind': 'colab_1b_v1', 'base_total_positions': base_positions,
                'additional_nonpadding_positions': total, 'additional_loss_positions': loss_total,
                'approx_total_positions': base_positions + total, 'colab_steps': step,
                'target_additional_positions': args.target_tokens, 'batch_size': args.batch_size,
                'learning_rate': args.lr, 'dataset_sha256': fingerprint, 'source_counts': counts,
                'base_checkpoint': args.base.name, 'heldout_evaluation_complete': False,
                'warning': 'Repeated unreviewed examples; prompts included in input count but not QA loss.'}
        payload = {'config': model.cfg.to_dict(), 'state_dict': model.state_dict(),
                   'optimizer': opt.state_dict(), 'metadata': meta,
                   'python_random_state': rng.getstate(), 'torch_rng_state': torch.get_rng_state(),
                   'cuda_rng_state': torch.cuda.get_rng_state_all(), 'scaler': scaler.state_dict()}
        path = save_checkpoint(args.drive_dir, payload)
        print(f'Saved {path}; approx total {base_positions+total:,} positions', flush=True)

    try:
        while total < args.target_tokens:
            x, y, f, n, nl, _ = batch(examples, feat, tok.pad_id, rng, args.batch_size)
            x, y, f = x.to(device), y.to(device), f.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type='cuda', dtype=dtype):
                _, loss, _ = model(x, f, targets=y)
            if not torch.isfinite(loss):
                raise RuntimeError('Non-finite loss; stopping without saving invalid weights')
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scale_before = scaler.get_scale()
            scaler.step(opt)
            scaler.update()
            if scaler.is_enabled() and scaler.get_scale() < scale_before:
                print('Skipped overflowing update; positions not counted.', flush=True)
                continue
            total += n; loss_total += nl; step += 1
            if step % 100 == 0:
                print(f'step={step} loss={float(loss):.4f} additional={total:,}/{args.target_tokens:,} '
                      f'loss_positions={loss_total:,}', flush=True)
            if time.monotonic() - last_save_time >= args.save_seconds:
                save(); last_saved = total; last_save_time = time.monotonic()
    except KeyboardInterrupt:
        if total != last_saved:
            save()
        print('Interrupted; resume by rerunning the notebook.', flush=True)
        return
    if total != last_saved:
        save()
    print(f'Target reached: {total:,} additional non-padding input positions.', flush=True)


if __name__ == '__main__':
    main()
