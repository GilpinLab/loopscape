"""Transformer building blocks (RoPE attention, SwiGLU, RMSNorm, casted layers).

Reconstructed for inference from the public TRM/HRM ``models/layers.py`` that
FPRM builds on.  Two differences from the upstream file:

* ``CastedLinear`` carries a ``mask`` buffer (weight-sparsity mask).  In the
  released FPRM Sudoku checkpoint every mask is all-ones, so ``weight * mask``
  is a no-op, but we keep the buffer so ``load_state_dict`` matches exactly.
* Attention uses ``torch.nn.functional.scaled_dot_product_attention`` instead of
  FlashAttention so it runs on CPU.
"""
from typing import Tuple

import einops
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.functional import scaled_dot_product_attention

from .common import trunc_normal_init_

CosSin = Tuple[torch.Tensor, torch.Tensor]


def _find_multiple(a, b):
    return (-(a // -b)) * b


def rotate_half(x: torch.Tensor):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    # q, k: [bs, seq_len, num_heads, head_dim]; cos, sin: [seq_len, head_dim]
    # Cast the (float64) RoPE cache *down* to the activation dtype rather than
    # forcing q/k up to the cache's dtype.  The old direction pinned q/k to the
    # cache's float32, which -- like the rms_norm cast -- silently capped an fp64
    # recurrence at fp32 inside attention.
    orig_dtype = q.dtype
    cos = cos.to(q.dtype)
    sin = sin.to(q.dtype)

    q_embed = (q * cos.unsqueeze(-2)) + (rotate_half(q) * sin.unsqueeze(-2))
    k_embed = (k * cos.unsqueeze(-2)) + (rotate_half(k) * sin.unsqueeze(-2))

    return q_embed.to(orig_dtype), k_embed.to(orig_dtype)


class CastedLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool):
        super().__init__()
        self.weight = nn.Parameter(
            trunc_normal_init_(torch.empty((out_features, in_features)), std=1.0 / (in_features ** 0.5))
        )
        self.bias = None
        if bias:
            self.bias = nn.Parameter(torch.zeros((out_features,)))
        # Weight-sparsity mask (all-ones in the released checkpoint -> no-op).
        self.register_buffer("mask", torch.ones((out_features, in_features)), persistent=True)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        weight = (self.weight * self.mask).to(input.dtype)
        bias = self.bias.to(input.dtype) if self.bias is not None else None
        return F.linear(input, weight, bias=bias)


class CastedEmbedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, init_std: float, cast_to: torch.dtype):
        super().__init__()
        self.cast_to = cast_to
        self.embedding_weight = nn.Parameter(
            trunc_normal_init_(torch.empty((num_embeddings, embedding_dim)), std=init_std)
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.embedding(input, self.embedding_weight.to(self.cast_to))


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings, base, device=None):
        super().__init__()
        # Build the cache in float64 and let ``apply_rotary_pos_emb`` cast it down
        # to whatever dtype the activations use.  Built in float32 (as upstream
        # did) the cache became the precision ceiling for q/k in attention.
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float64, device=device) / dim))
        t = torch.arange(max_position_embeddings, dtype=torch.float64, device=device)
        freqs = torch.outer(t, inv_freq)

        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self):
        return self.cos_cached, self.sin_cached


class Attention(nn.Module):
    def __init__(self, hidden_size, head_dim, num_heads, num_key_value_heads, causal=False):
        super().__init__()
        self.hidden_size = hidden_size
        self.head_dim = head_dim
        self.output_size = head_dim * num_heads
        self.num_heads = num_heads
        self.num_key_value_heads = num_key_value_heads
        self.causal = causal

        self.qkv_proj = CastedLinear(self.hidden_size, (self.num_heads + 2 * self.num_key_value_heads) * self.head_dim, bias=False)
        self.o_proj = CastedLinear(self.output_size, self.hidden_size, bias=False)

    def forward(self, cos_sin: CosSin, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape

        qkv = self.qkv_proj(hidden_states)
        qkv = qkv.view(batch_size, seq_len, self.num_heads + 2 * self.num_key_value_heads, self.head_dim)
        query = qkv[:, :, :self.num_heads]
        key = qkv[:, :, self.num_heads: self.num_heads + self.num_key_value_heads]
        value = qkv[:, :, self.num_heads + self.num_key_value_heads:]

        if cos_sin is not None:
            cos, sin = cos_sin
            query, key = apply_rotary_pos_emb(query, key, cos, sin)

        query, key, value = map(lambda t: einops.rearrange(t, 'B S H D -> B H S D'), (query, key, value))
        attn_output = scaled_dot_product_attention(query=query, key=key, value=value, is_causal=self.causal)
        attn_output = einops.rearrange(attn_output, 'B H S D -> B S H D')
        attn_output = attn_output.reshape(batch_size, seq_len, self.output_size)
        return self.o_proj(attn_output)


class SwiGLU(nn.Module):
    def __init__(self, hidden_size: int, expansion: float):
        super().__init__()
        inter = _find_multiple(round(expansion * hidden_size * 2 / 3), 256)

        self.gate_up_proj = CastedLinear(hidden_size, inter * 2, bias=False)
        self.down_proj = CastedLinear(inter, hidden_size, bias=False)

    def forward(self, x):
        gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        return self.down_proj(F.silu(gate) * up)


def rms_norm(hidden_states: torch.Tensor, variance_epsilon: float) -> torch.Tensor:
    """RMS-norm, computed in at least float32.

    Upstream hardcoded ``.to(torch.float32)`` here.  That is the right thing for
    fp16/bf16 (the variance sum needs the headroom) but it silently *caps the whole
    recurrence at fp32*: the maze config applies this norm five times per loop
    iteration (two pre-norms per layer x L_layers=2, plus norm_placement="output"),
    so a float64 latent was being rounded to float32 five times per step.  That
    makes genuine fp64 trajectories impossible, which matters for precision-
    sensitive work -- Lyapunov exponents, fp32-vs-fp64 shadowing checks, and
    fractal-boundary renders all silently measured fp32 rounding instead.

    Preserving float64 while still upcasting the low-precision dtypes leaves
    fp32/bf16 behaviour bit-identical to before.
    """
    input_dtype = hidden_states.dtype
    compute_dtype = (torch.float32
                     if input_dtype in (torch.float16, torch.bfloat16)
                     else input_dtype)
    hidden_states = hidden_states.to(compute_dtype)

    variance = hidden_states.square().mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + variance_epsilon)
    return hidden_states.to(input_dtype)
