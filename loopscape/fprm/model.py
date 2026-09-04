"""FPRM single-Z fixed-point reasoning model (inference path).

Adapted from the release artifact's ``fp_trm_singlez.py`` with local imports and the
training-only code paths removed.  Parameter names are unchanged so the released
checkpoint loads exactly.
"""
import math
from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from .config import FPTRMConfig
from .fixed_point import FixedPointOptimizer
from .layers import CastedEmbedding, CastedLinear, RotaryEmbedding
from .sparse_embedding import CastedSparseEmbedding
from .transformer import FixedPointTransformer

IGNORE_LABEL_ID = -100


@dataclass
class InnerCarry:
    z_L_state: dict
    dropout_mask: torch.Tensor


@dataclass
class Carry:
    inner_carry: InnerCarry
    steps: torch.Tensor
    halted: torch.Tensor
    current_data: Dict[str, torch.Tensor]


class FPTRMInner(nn.Module):
    def __init__(self, config: FPTRMConfig) -> None:
        super().__init__()
        self.config = config
        self.forward_dtype = getattr(torch, self.config.forward_dtype)

        self.embed_scale = math.sqrt(self.config.hidden_size)
        embed_init_std = 1.0 / self.embed_scale

        self.embed_tokens = CastedEmbedding(self.config.vocab_size, self.config.hidden_size,
                                            init_std=embed_init_std, cast_to=self.forward_dtype)
        self.lm_head = CastedLinear(self.config.hidden_size, self.config.vocab_size, bias=False)
        self.q_head = CastedLinear(self.config.hidden_size, 2, bias=True)

        self.puzzle_emb_len = (
            -(self.config.puzzle_emb_ndim // -self.config.hidden_size)
            if self.config.puzzle_emb_len == 0 else self.config.puzzle_emb_len
        )
        if self.config.puzzle_emb_ndim > 0:
            self.puzzle_emb = CastedSparseEmbedding(
                self.config.num_puzzle_identifiers, self.config.puzzle_emb_ndim,
                batch_size=self.config.batch_size, init_std=0, cast_to=self.forward_dtype,
            )

        if self.config.pos_encodings == "rope":
            self.rotary_emb = RotaryEmbedding(
                dim=self.config.hidden_size // self.config.num_heads,
                max_position_embeddings=self.config.seq_len + self.puzzle_emb_len,
                base=self.config.rope_theta,
            )
        elif self.config.pos_encodings == "learned":
            self.embed_pos = CastedEmbedding(self.config.seq_len + self.puzzle_emb_len,
                                             self.config.hidden_size, init_std=embed_init_std,
                                             cast_to=self.forward_dtype)

        self.L_level = FixedPointTransformer(self.config, self.config.L_layers)
        self.L_optimizer = FixedPointOptimizer(self.config)

    def _input_embeddings(self, input: torch.Tensor, puzzle_identifiers: torch.Tensor):
        embedding = self.embed_tokens(input.to(torch.int32))

        if self.config.puzzle_emb_ndim > 0:
            puzzle_embedding = self.puzzle_emb(puzzle_identifiers)
            pad_count = self.puzzle_emb_len * self.config.hidden_size - puzzle_embedding.shape[-1]
            if pad_count > 0:
                puzzle_embedding = F.pad(puzzle_embedding, (0, pad_count))
            embedding = torch.cat(
                (puzzle_embedding.view(-1, self.puzzle_emb_len, self.config.hidden_size), embedding), dim=-2)

        if self.config.pos_encodings == "learned":
            embedding = 0.707106781 * (embedding + self.embed_pos.embedding_weight.to(self.forward_dtype))

        return self.embed_scale * embedding

    def empty_carry(self, batch_size: int):
        return InnerCarry(z_L_state=None, dropout_mask=None)

    def reset_carry(self, reset_flag: torch.Tensor, batch: torch.Tensor, carry: InnerCarry):
        shape = (batch.shape[0], batch.shape[1] + self.puzzle_emb_len, self.config.hidden_size)
        device = batch.device
        dtype = self.forward_dtype
        # Inference: dropout mask is all-ones.
        dropout_mask = torch.ones(*shape, device=device, dtype=dtype)
        return InnerCarry(
            z_L_state=self.L_optimizer.reset(reset_flag, shape, dtype, device, carry.z_L_state),
            dropout_mask=dropout_mask,
        )

    def _z_step(self, state, input_embeddings, dropout_mask, seq_info):
        z_new = dropout_mask * self.L_level(state["y"], input_embeddings, **seq_info)
        return self.L_optimizer.step(state, z_new)

    def forward(self, carry: InnerCarry, batch: Dict[str, torch.Tensor], n_steps: int):
        input_embeddings = self._input_embeddings(batch["inputs"], batch["puzzle_identifiers"])

        cos_sin = None
        if hasattr(self, "rotary_emb"):
            cos, sin = self.rotary_emb()
            s = input_embeddings.shape[1]
            cos_sin = (cos[:s], sin[:s])
        seq_info = dict(cos_sin=cos_sin, puzzle_emb_len=self.puzzle_emb_len)
        z_state, dropout_mask = carry.z_L_state, carry.dropout_mask

        with torch.no_grad():
            for _ in range(n_steps):
                z_state = self._z_step(z_state, input_embeddings, dropout_mask, seq_info)

        pred_rep = z_state["y"]
        for _ in range(self.config.n_decode_steps):
            pred_rep = self.L_level(pred_rep, z_state["y"], **seq_info)

        output = self.lm_head(pred_rep)[:, self.puzzle_emb_len:]
        q_logits = self.q_head(pred_rep[:, 0]).to(torch.float32)
        new_carry = InnerCarry(z_L_state=self.L_optimizer.detach_state(z_state),
                               dropout_mask=carry.dropout_mask)
        return new_carry, output, (q_logits[..., 0], q_logits[..., 1])


class FPTRM(nn.Module):
    """Single-state FPRM wrapper (inference)."""

    def __init__(self, config_dict: dict):
        super().__init__()
        self.config = FPTRMConfig(**config_dict)
        self.inner = FPTRMInner(self.config)

    @property
    def puzzle_emb(self):
        return self.inner.puzzle_emb

    def initial_carry(self, batch: Dict[str, torch.Tensor]):
        batch_size = batch["inputs"].shape[0]
        # steps/halted must live on the batch device: they feed torch.where in
        # reset_carry, which rejects CPU flags against CUDA state.
        device = batch["inputs"].device
        return Carry(
            inner_carry=self.inner.empty_carry(batch_size),
            steps=torch.zeros((batch_size,), dtype=torch.int32, device=device),
            halted=torch.ones((batch_size,), dtype=torch.bool, device=device),
            current_data={k: torch.empty_like(v) for k, v in batch.items()},
        )

    @property
    def max_iter(self):
        return self.config.max_iter_eval if self.config.max_iter_eval is not None else self.config.max_iter

    def forward(self, carry: Carry, batch: Dict[str, torch.Tensor]) -> Tuple[Carry, Dict[str, torch.Tensor]]:
        new_inner_carry = self.inner.reset_carry(carry.halted, batch["inputs"], carry.inner_carry)
        new_steps = torch.where(carry.halted, 0, carry.steps)
        new_current_data = {
            k: torch.where(carry.halted.view((-1,) + (1,) * (batch[k].ndim - 1)), batch[k], v)
            for k, v in carry.current_data.items()
        }

        new_inner_carry, logits, (q_halt_logits, q_continue_logits) = self.inner(
            new_inner_carry, new_current_data, n_steps=1)

        outputs = {
            "logits": logits,
            "q_halt_logits": q_halt_logits,
            "q_continue_logits": q_continue_logits,
        }

        with torch.no_grad():
            new_steps = new_steps + 1
            is_last_step = new_steps >= self.max_iter
            halted = is_last_step
            # Fixed-point halting: stop once the residual is below tau (fp_thresh)
            # for the whole batch, or the step size collapses.
            halted = halted | (new_inner_carry.z_L_state["residues"].max() < self.config.fp_thresh) \
                            | (new_inner_carry.z_L_state["stepsize"].max() < 1e-3)

        return Carry(new_inner_carry, new_steps, halted, new_current_data), outputs
