"""Answer and settle-time fields over a 2-D slice of the initial loop state.

The slice is a base point on the shell ``||z|| = R = 0.1 sqrt(n d)`` (the scale of the model's
own random init) plus two orthonormal directions, all on the answer slots.  The settle time
of a point is the last iteration at which its decoded answer changed.
"""
from __future__ import annotations

import numpy as np
import torch

from .download import ensure_linsys_checkpoint
from .model import LoopedTransformer
from .task import N, P, answer_code, encode


def load(path: str | None = None, device: str = "cpu") -> LoopedTransformer:
    return LoopedTransformer.from_checkpoint(path or ensure_linsys_checkpoint(), device)


def plane(seq_len, d, R, seed=0, device="cpu"):
    """Base point ``z0`` on the shell and orthonormal ``u1, u2``."""
    g = torch.Generator(device=device).manual_seed(int(seed))

    def draw():
        v = torch.zeros(1, seq_len, d, device=device, dtype=torch.float64)
        v[0, seq_len - N:] = torch.randn(N, d, generator=g, device=device, dtype=torch.float64)
        return v

    z0 = draw(); z0 = z0 / z0.norm() * R
    u1 = draw(); u1 = u1 / u1.norm()
    u2 = draw(); u2 = u2 - (u2 * u1).sum() * u1; u2 = u2 / u2.norm()
    return z0.float(), u1.float(), u2.float()


@torch.no_grad()
def settle_field(model, A, b, res=128, half_width=2.0, loops=320, seed=29, batch=1024):
    """``(answers, settle)`` over ``z0 + R (a u1 + c u2)`` for ``a, c`` in ``[-half_width, half_width]``."""
    device = next(model.parameters()).device
    idx = torch.tensor(encode(A, b), device=device).unsqueeze(0)
    T = idx.shape[1]
    R = 0.1 * float(np.sqrt(N * model.config.n_embd))
    z0, u1, u2 = plane(T, model.config.n_embd, R, seed, device)
    xs = np.linspace(-half_width, half_width, res)
    aa, cc = (torch.tensor(v.ravel(), dtype=torch.float32, device=device)
              for v in np.meshgrid(xs, xs, indexing="ij"))
    answers = np.empty(res * res, dtype=np.int64)
    settle = np.empty(res * res, dtype=np.int64)
    for s in range(0, res * res, batch):
        e = min(s + batch, res * res)
        z = z0 + R * (aa[s:e].view(-1, 1, 1) * u1 + cc[s:e].view(-1, 1, 1) * u2)
        dec = []
        model(idx.expand(e - s, -1), loops=loops, init_x=z,
              on_iter=lambda k, x: dec.append(model.decode(x[:, T - N:]).argmax(-1)))
        D = torch.stack(dec) % P                                   # (loops, B, N)
        changed = (D[1:] != D[:-1]).any(-1).cpu().numpy()
        settle[s:e] = np.where(changed.any(0), len(changed) - np.argmax(changed[::-1], 0), 0)
        answers[s:e] = answer_code(D[-1].cpu().numpy())
    return answers.reshape(res, res), settle.reshape(res, res)
