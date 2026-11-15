"""Training loop for the Sanskrit LM (CPU friendly, resumable, logged)."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import torch

from .data import TokenDataset
from .model import SanskritLM, save_checkpoint


def lr_at(step: int, *, lr: float, warmup: int, total: int, min_lr_ratio: float = 0.1) -> float:
    if step < warmup:
        return lr * (step + 1) / max(1, warmup)
    progress = min(1.0, (step - warmup) / max(1, total - warmup))
    cos = 0.5 * (1 + math.cos(math.pi * progress))
    return lr * (min_lr_ratio + (1 - min_lr_ratio) * cos)


@torch.no_grad()
def evaluate(model: SanskritLM, dataset: TokenDataset, *, batches: int = 8,
             batch_size: int = 8) -> float:
    model.eval()
    total, count = 0.0, 0
    for i, (x, y, f) in enumerate(dataset.sequential_batches(batch_size, stride=dataset.block_size)):
        if i >= batches:
            break
        _, loss, _ = model(x, f, targets=y)
        total += float(loss)
        count += 1
    model.train()
    return total / max(count, 1)


def train(
    model: SanskritLM,
    train_ds: TokenDataset,
    val_ds: TokenDataset | None,
    tokenizer,
    *,
    steps: int = 1000,
    batch_size: int = 16,
    lr: float = 1.5e-3,
    warmup: int = 100,
    weight_decay: float = 0.1,
    grad_clip: float = 1.0,
    log_every: int = 25,
    eval_every: int = 250,
    ckpt_every: int = 500,
    out_dir: Path = Path("artifacts"),
    resume: Path | None = None,
    reset_steps: bool = False,
    threads: int = 2,
) -> dict:
    torch.set_num_threads(threads)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    decay, no_decay = [], []
    for name, p in model.named_parameters():
        (decay if p.dim() >= 2 else no_decay).append(p)
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": weight_decay},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=lr, betas=(0.9, 0.95), eps=1e-8)

    start_step = 0
    if resume and Path(resume).exists():
        blob = torch.load(resume, map_location="cpu", weights_only=False)
        model.load_state_dict(blob["state_dict"])
        # a checkpoint rebuilt from the deployed NumPy bundle carries no optimizer
        # state; the weights are all that matter when the LR schedule restarts
        if blob.get("optimizer"):
            opt.load_state_dict(blob["optimizer"])
        start_step = 0 if reset_steps else int(blob.get("step", 0))
        print(f"resumed from {resume} at step {start_step}"
              + (" (step counter reset)" if reset_steps else ""))

    model.train()
    history: list[dict] = []
    t_start = time.time()
    tokens_seen = 0
    running_loss = 0.0
    best_val = float("inf")
    metrics_path = out_dir / "metrics.json"

    for step in range(start_step, steps):
        cur_lr = lr_at(step, lr=lr, warmup=warmup, total=steps)
        for g in opt.param_groups:
            g["lr"] = cur_lr

        x, y, f = train_ds.sample_batch(batch_size)
        _, loss, _ = model(x, f, targets=y)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()

        running_loss += float(loss.detach())
        tokens_seen += x.numel()

        if step % log_every == 0:
            avg = running_loss / max(1, (step - start_step) // log_every + 1)
            elapsed = time.time() - t_start
            tok_s = tokens_seen / max(elapsed, 1e-9)
            print(f"step {step:5d}/{steps}  loss {float(loss.detach()):.4f}  "
                  f"lr {cur_lr:.2e}  {tok_s:6.0f} tok/s  "
                  f"{elapsed / 60:.1f} min", flush=True)
            history.append({
                "step": step, "loss": round(float(loss), 4), "lr": round(cur_lr, 6),
                "tok_per_s": round(tok_s, 1),
                "elapsed_s": round(elapsed, 1),
            })

        if val_ds is not None and eval_every and (step + 1) % eval_every == 0:
            vloss = evaluate(model, val_ds)
            ppl = math.exp(min(vloss, 20))
            print(f"  >> eval @ {step + 1}: val loss {vloss:.4f}  ppl {ppl:.1f}",
                  flush=True)
            history.append({"step": step + 1, "val_loss": round(vloss, 4),
                            "val_ppl": round(ppl, 1)})
            if vloss < best_val:
                best_val = vloss
                save_checkpoint(out_dir / "best.pt", model, tokenizer,
                                {"step": step + 1, "val_loss": vloss})
            metrics_path.write_text(json.dumps(history, indent=2), encoding="utf-8")

        if ckpt_every and (step + 1) % ckpt_every == 0:
            torch.save({
                "config": model.cfg.to_dict(),
                "state_dict": model.state_dict(),
                "optimizer": opt.state_dict(),
                "step": step + 1,
            }, out_dir / "latest.pt")
            save_checkpoint(out_dir / "model.pt", model, tokenizer,
                            {"step": step + 1, "loss": float(loss)})

    save_checkpoint(out_dir / "model.pt", model, tokenizer,
                    {"step": steps, "final": True})
    metrics_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    summary = {
        "steps": steps,
        "batch_size": batch_size,
        "block_size": train_ds.block_size,
        "tokens_seen": tokens_seen,
        "params": model.num_params(),
        "minutes": round((time.time() - t_start) / 60, 2),
        "best_val_loss": None if best_val == float("inf") else round(best_val, 4),
        "final_loss": round(float(loss), 4),
    }
    (out_dir / "train_summary.json").write_text(json.dumps(summary, indent=2),
                                                encoding="utf-8")
    return summary
