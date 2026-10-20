"""Looped transformer: one pre-norm block applied ``loops`` times.

The token embeddings ``e`` are re-injected every iteration and the loop state starts from
noise on the answer slots::

    x_0 = init_std * randn * slot_mask
    x_{k+1} = LN(x_k + step_size * (block(x_k + e) - (x_k + e)))
    logits = lm_head(ln_f(x_loops))
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F

from .task import SLOT


@dataclass
class Config:
    vocab_size: int = 32
    block_size: int = 256
    n_embd: int = 256
    n_head: int = 8
    mlp_ratio: float = 1.0
    default_loops: int = 16
    max_grad_loops: int = 8
    step_size: float = 1.0
    init_std: float = 0.1


def _rope(x, cos, sin):
    x_ = x.float().reshape(*x.shape[:-1], -1, 2)
    x0, x1 = x_[..., 0], x_[..., 1]
    c, s = cos[: x.shape[-2]][None, None], sin[: x.shape[-2]][None, None]
    return torch.stack((x0 * c - x1 * s, x0 * s + x1 * c), dim=-1).flatten(3).type_as(x)


class Attention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n_head = cfg.n_head
        self.c_attn = nn.Linear(cfg.n_embd, 3 * cfg.n_embd)
        self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd)
        d = cfg.n_embd // cfg.n_head
        freqs = 1.0 / (10000.0 ** (torch.arange(0, d, 2).float() / d))
        ang = torch.outer(torch.arange(cfg.block_size).float(), freqs)
        self.register_buffer("cos", torch.cos(ang), persistent=False)
        self.register_buffer("sin", torch.sin(ang), persistent=False)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = (t.view(B, T, self.n_head, -1).transpose(1, 2) for t in self.c_attn(x).split(C, dim=2))
        y = F.scaled_dot_product_attention(_rope(q, self.cos, self.sin), _rope(k, self.cos, self.sin), v)
        return self.c_proj(y.transpose(1, 2).reshape(B, T, C))


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        hidden = int(round(cfg.mlp_ratio * cfg.n_embd))
        self.ln_1, self.attn = nn.LayerNorm(cfg.n_embd), Attention(cfg)
        self.ln_2 = nn.LayerNorm(cfg.n_embd)
        self.mlp = nn.Sequential(nn.Linear(cfg.n_embd, hidden), nn.GELU(), nn.Linear(hidden, cfg.n_embd))

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        return x + self.mlp(self.ln_2(x))


class LoopedTransformer(nn.Module):
    def __init__(self, cfg: Config = Config()):
        super().__init__()
        self.config = cfg
        self.wte = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.block = Block(cfg)
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.loop_ln = nn.LayerNorm(cfg.n_embd, bias=False)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.wte.weight = self.lm_head.weight
        for name, p in self.named_parameters():
            if name.endswith("bias"):
                nn.init.zeros_(p)
            elif p.dim() == 2:
                std = 0.02 / math.sqrt(2 * cfg.default_loops) if name in ("block.attn.c_proj.weight",
                                                                         "block.mlp.2.weight") else 0.02
                nn.init.normal_(p, std=std)
        # so ||x|| starts comparable to ||e|| rather than sqrt(n_embd)
        nn.init.constant_(self.loop_ln.weight, 0.02)

    def step(self, x, e):
        u = x + e
        return self.loop_ln(x + self.config.step_size * (self.block(u) - u))

    def decode(self, x):
        return self.lm_head(self.ln_f(x))

    def forward(self, idx, loops=None, init_x=None, on_iter=None):
        """Logits after ``loops`` iterations; only the last ``max_grad_loops`` carry gradient.

        ``init_x`` replaces the random initial state; ``on_iter(k, x)`` sees every state.
        """
        loops = loops or self.config.default_loops
        e = self.wte(idx)
        if init_x is None:
            x = self.config.init_std * torch.randn_like(e) * (idx == SLOT).unsqueeze(-1).to(e.dtype)
        else:
            x = init_x.expand_as(e)
        grad = torch.is_grad_enabled()
        for k in range(loops):
            with torch.set_grad_enabled(grad and k >= loops - self.config.max_grad_loops):
                x = self.step(x, e)
            if on_iter is not None:
                on_iter(k + 1, x)
        return self.decode(x)

    @classmethod
    def from_checkpoint(cls, path, device="cpu"):
        sd = torch.load(path, map_location="cpu", weights_only=True)
        rename = {"transformer.": "", "block.block.": "block.", "mlp.c_fc.": "mlp.0.", "mlp.c_proj.": "mlp.2."}
        for a, b in rename.items():  # checkpoints from the original training code
            sd = {k.replace(a, b): v for k, v in sd.items()}
        sd.pop("block.attn.bias", None)  # unused causal mask
        m = cls()
        m.load_state_dict(sd)
        return m.to(device).eval()
