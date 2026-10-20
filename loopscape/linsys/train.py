"""Train the looped transformer with the released checkpoint's recipe (~4 h on one H100).

    python -m loopscape.linsys.train --out runs/linsys

Checkpoints load with ``basin.load(path)``.
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np
import torch
from torch.nn import functional as F

from .model import LoopedTransformer
from .task import N, PAD, sample_batch


class _Batches(torch.utils.data.IterableDataset):
    def __init__(self, batch_size, seed):
        self.batch_size, self.seed = batch_size, seed

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        rng = np.random.default_rng([self.seed, info.id if info else 0])
        while True:
            yield sample_batch(self.batch_size, rng)


def loop_count(rng, mean=16, sigma=0.3):
    """Poisson with a log-normal rate whose mean is pinned at ``mean``."""
    tau = sigma * rng.normal() + np.log(mean) - 0.5 * sigma ** 2
    return max(1, int(rng.poisson(np.exp(tau))))


def lr_factor(step, n_steps, warmup=2000, cooldown_frac=0.05):
    """Warmup, constant, then a 1-sqrt decay to zero over the last ``cooldown_frac``."""
    decay_start = max(warmup, n_steps - max(1, int(cooldown_frac * n_steps)))
    if step < warmup:
        return max(1e-4, step / warmup)
    if step < decay_start:
        return 1.0
    return max(0.0, 1.0 - ((step - decay_start) / max(1, n_steps - decay_start)) ** 0.5)


@torch.no_grad()
def solve_rate(model, eval_set, loops=16):
    """Fraction of held-out instances whose whole answer is correct."""
    model.eval()
    x, y = eval_set
    hits = sum((model(xb, loops=loops)[:, -N:].argmax(-1) == yb[:, -N:]).all(-1).sum().item()
               for xb, yb in zip(x.split(512), y.split(512)))
    model.train()
    return hits / len(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=300000)
    ap.add_argument("--batch_size", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-3)
    ap.add_argument("--save_every", type=int, default=2000)
    ap.add_argument("--eval_every", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--seed", type=int, default=2)
    ap.add_argument("--no_compile", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    torch.manual_seed(a.seed)

    model = LoopedTransformer().cuda().train()
    if not a.no_compile:
        model.step = torch.compile(model.step)
    ps = list(model.parameters())
    opt = torch.optim.AdamW([{"params": [p for p in ps if p.dim() >= 2], "weight_decay": a.weight_decay},
                             {"params": [p for p in ps if p.dim() < 2], "weight_decay": 0.0}],
                            lr=a.lr, betas=(0.9, 0.99))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: lr_factor(s, a.steps))
    data = iter(torch.utils.data.DataLoader(_Batches(a.batch_size, a.seed), batch_size=None,
                                            num_workers=a.workers, prefetch_factor=4))
    rng = np.random.default_rng(a.seed)
    # built once: only 5% of draws land on the test side of the split
    eval_set = tuple(t.cuda() for t in sample_batch(2048, np.random.default_rng(a.seed + 1), "test"))
    t0 = time.time()
    for step in range(a.steps):
        x, y = (t.cuda(non_blocking=True) for t in next(data))
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(x, loops=loop_count(rng))
            mask = y != PAD
            loss = F.cross_entropy(logits[mask].float(), y[mask])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if (step + 1) % a.eval_every == 0:
            print(f"step {step + 1}  loss {loss.item():.4f}  solved {solve_rate(model, eval_set):.3f}"
                  f"  {time.time() - t0:.0f}s", flush=True)
        if (step + 1) % a.save_every == 0 or step + 1 == a.steps:
            torch.save(model.state_dict(), os.path.join(a.out, f"model_{step + 1}.pt"))


if __name__ == "__main__":
    main()
