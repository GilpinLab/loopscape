"""TRM solver -- Tiny Recursive Model (arXiv:2510.04871).

TRM is the HRM/EqR-family recursive reasoner distilled to its essence: a single
tiny 2-layer network ``L_level`` updates both latents -- ``z_L`` (the "reasoning"
latent, ``z`` in the paper) against the input injection, then ``z_H`` (the
"answer" latent, ``y``) against ``z_L`` -- for ``H_cycles x L_cycles`` inner
iterations per ACT outer step.  Unlike EqR it is **fully deterministic**: latents
reset to fixed trained ``H_init``/``L_init`` buffers (no random init, no per-step
noise), so there is no ``seed`` knob and initial-condition sweeps are specified
purely via injected ``z_H``/``z_L``.  Eval-mode ACT halting is a fixed
``halt_max_steps`` (16), exactly like EqR; the learned halting signal is exposed
as ``extra["q_halt_logits"]``.

Only the **maze** task has a public checkpoint: Samsung released no weights, so
``load()`` fetches alphaXiv's independent reproduction of the paper's Maze-Hard
recipe (83.67% exact accuracy vs the paper's 85.3%; see
:mod:`_trm_download`).  The latent shape is ``(B, 916, 512)`` = ``(B, seq_len
900 + puzzle_emb_len 16, hidden)`` -- the same family as EqR's maze latents.

The model code is vendored under ``loopscape/trm/_vendor`` (upstream files with
imports made package-relative; see that package's docstring).
"""
from __future__ import annotations

from typing import List, Optional, Union

import torch

from ..puzzle import Grid
from ..tasks import get_task
from ..base import BatchSolveResult, Solver, SolveResult, register_solver
from .download import ensure_trm_checkpoint

# Architecture config for the Maze-Hard TRM checkpoint.  Values match the
# upstream README's maze recipe (and the all_config.yaml shipped with the
# Sanjin2024 reproduction, which alphaXiv's run also follows): attention blocks
# (mlp_t off), rope, one shared 2-layer L_level, 3 H-cycles x 4 L-cycles per ACT
# step.  Dataset-derived fields: vocab 6 ("# SGo" + pad), seq_len 900 (30x30),
# a single puzzle identifier.
_TRM_MAZE_CONFIG = dict(
    batch_size=1,                 # only used by train-mode sparse-embedding buffers
    seq_len=900,
    vocab_size=6,
    num_puzzle_identifiers=1,
    H_cycles=3,
    L_cycles=4,
    H_layers=0,                   # ignored by TRM (single shared L_level)
    L_layers=2,
    hidden_size=512,
    expansion=4,
    num_heads=8,
    pos_encodings="rope",
    halt_max_steps=16,
    halt_exploration_prob=0.1,
    forward_dtype="bfloat16",
    mlp_t=False,
    puzzle_emb_len=16,
    puzzle_emb_ndim=512,
)

_TRM_CONFIG = {"maze": _TRM_MAZE_CONFIG}

_PREFIX = "model."   # checkpoint keys are ACTLossHead-wrapped: model.inner.*


def _rms_update(z_new: torch.Tensor, z_old: torch.Tensor) -> torch.Tensor:
    """Per-sample RMS latent update ||z_new - z_old|| -- the residual analogue,
    same definition as the EqR solver's."""
    return (z_new.float() - z_old.float()).pow(2).mean(dim=(1, 2)).sqrt()


def _extract_state_dict(ckpt) -> dict:
    """Strip the ACTLossHead ``model.`` prefix from a TRM training checkpoint."""
    sd = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt and not hasattr(ckpt["model"], "shape") else ckpt
    return {k[len(_PREFIX):] if k.startswith(_PREFIX) else k: v for k, v in sd.items()}


def _to_device(carry, device):
    """initial_carry builds CPU tensors; move every piece to the model's device."""
    carry.inner_carry.z_H = carry.inner_carry.z_H.to(device)
    carry.inner_carry.z_L = carry.inner_carry.z_L.to(device)
    carry.steps = carry.steps.to(device)
    carry.halted = carry.halted.to(device)
    carry.current_data = {k: v.to(device) for k, v in carry.current_data.items()}
    return carry


def _latent_shape(model, batch) -> tuple:
    B = batch["inputs"].shape[0]
    return (B, model.config.seq_len + model.inner.puzzle_emb_len,
            model.config.hidden_size)                       # (B, 916, 512) for maze


def _prepare_latent(z, model, batch, name: str):
    """Validate/broadcast a user-supplied z_H or z_L to (B, seq_len+16, hidden)."""
    device = next(model.parameters()).device
    dtype = model.inner.forward_dtype
    expected = _latent_shape(model, batch)
    t = torch.as_tensor(z, device=device, dtype=dtype)
    if t.ndim == 2:
        t = t.unsqueeze(0)
    if tuple(t.shape) != expected:
        raise ValueError(
            f"{name} must have shape {expected} or {expected[1:]}, got {tuple(t.shape)}"
        )
    return t


def _prime_trm_carry(model, batch, z_H=None, z_L=None):
    """Build a carry that starts from user-supplied z_H/z_L instead of H_init/L_init.

    Mirrors the EqR solver's carry-priming: normally the first forward resets
    both latents (every sample starts ``halted``) to the fixed trained
    ``H_init``/``L_init`` buffers.  Setting the latents and clearing ``halted``
    makes the model keep them; ``current_data`` is seeded with the real puzzle
    since we skip the halted reset that normally fills it.  A latent left as
    ``None`` keeps its default ``H_init``/``L_init`` reset.
    """
    device = next(model.parameters()).device
    carry = _to_device(model.initial_carry(batch), device)
    if z_H is None and z_L is None:
        return carry

    # Reset both latents to their trained defaults first, so injecting only one
    # leaves the other well-defined (not uninitialised empty memory).
    reset_all = torch.ones_like(carry.halted)
    carry.inner_carry = model.inner.reset_carry(reset_all, carry.inner_carry)

    if z_H is not None:
        carry.inner_carry.z_H = _prepare_latent(z_H, model, batch, "z_H")
    if z_L is not None:
        carry.inner_carry.z_L = _prepare_latent(z_L, model, batch, "z_L")

    carry.steps = torch.zeros_like(carry.steps)
    carry.halted = torch.zeros_like(carry.halted)        # not halted -> keep latents
    carry.current_data = {k: v.clone() for k, v in batch.items()}
    return carry


class TRMSolver(Solver):
    """Tiny Recursive Model -- deterministic, ACT-halting recursive solver."""

    name = "trm"

    def __init__(self, model, device: str = "cpu", task="maze"):
        super().__init__(model, device=device)
        self.task = get_task(task)

    # ------------------------------------------------------------------ loading
    @classmethod
    def load(
        cls,
        checkpoint: Optional[str] = None,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
        config_overrides: Optional[dict] = None,
        task: str = "maze",
        **kwargs,
    ) -> "TRMSolver":
        """Build the maze TRM and load the alphaXiv reproduction weights.

        ``task="maze"`` is the only task with public weights (Samsung released
        none; the Sudoku reproductions on the Hub use different arch variants).
        ``checkpoint`` may point to a local ``step_*``/``.pt`` state dict from
        the upstream trainer (e.g. your own run), which is used as-is.
        """
        from ._vendor.trm import TinyRecursiveReasoningModel_ACTV1

        t = get_task(task)
        if t.name not in _TRM_CONFIG:
            raise ValueError(f"TRM has no public checkpoint/config for task {t.name!r}; "
                             f"available: {sorted(_TRM_CONFIG)}")

        cfg = dict(_TRM_CONFIG[t.name])
        cfg["forward_dtype"] = {torch.float32: "float32", torch.bfloat16: "bfloat16",
                                torch.float16: "float16"}.get(dtype, "float32")
        if config_overrides:
            cfg.update(config_overrides)

        model = TinyRecursiveReasoningModel_ACTV1(cfg)

        ckpt_path = ensure_trm_checkpoint(checkpoint, task=t.name)
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state_dict = _extract_state_dict(ckpt)

        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        # Non-persistent buffers (rotary caches, sparse-embedding locals) may be absent.
        real_missing = [k for k in missing
                        if not any(s in k for s in ("cos_cached", "sin_cached",
                                                    "local_weights", "local_ids"))]
        if real_missing or unexpected:
            raise RuntimeError(
                f"TRM state dict mismatch.\n  missing: {real_missing}\n  unexpected: {unexpected}"
            )

        model = model.to(device=device, dtype=dtype)
        model.eval()
        return cls(model, device=device, task=t)

    # ------------------------------------------------------------------ solve
    @torch.no_grad()
    def solve(
        self,
        puzzle: Union[str, Grid],
        *,
        return_intermediates: bool = False,
        return_latents: bool = False,
        return_residuals: bool = False,
        max_steps: Optional[int] = None,
        z_H: Optional["torch.Tensor"] = None,
        z_L: Optional["torch.Tensor"] = None,
        **model_kwargs,
    ) -> SolveResult:
        """Solve a maze with TRM.

        TRM is **deterministic**: with default inits the result is a pure
        function of the puzzle, and with injected latents a pure function of
        ``(puzzle, z_H, z_L)`` -- there is no ``seed`` parameter.

        Parameters
        ----------
        z_H, z_L    : optional starting latents, each of shape
                      ``(seq_len + 16, hidden)`` = ``(916, 512)`` for maze (or
                      batched with a leading 1).  Either may be given
                      independently; the other keeps its trained
                      ``H_init``/``L_init`` reset.  ``z_H`` is the answer
                      ("y") latent that feeds the output head, ``z_L`` the
                      reasoning ("z") latent.
        max_steps   : override ``halt_max_steps`` (the fixed eval ACT loop
                      length, default 16).
        return_latents : record per-step latents in ``extra["z_H_history"]`` /
                      ``extra["z_L_history"]`` as float32 CPU tensors of shape
                      ``(steps, seq_len+16, hidden)``; also fills ``residuals``
                      with the per-step RMS update ``||z_H(t) - z_H(t-1)||``.
        return_residuals : fill ``residuals`` only, without storing the (large)
                      latent histories -- the cheap way to get convergence
                      times over big sweeps.

        ``extra["q_halt_logits"]`` carries the final learned halt logit (the
        model's own confidence signal; eval halting itself is the fixed cap).
        """
        if max_steps is not None:
            self.model.config.halt_max_steps = int(max_steps)
        track_res = return_latents or return_residuals

        device = next(self.model.parameters()).device
        inputs = self.task.encode(puzzle).to(device)                       # [1, seq_len]
        batch = {
            "inputs": inputs,
            "labels": torch.full_like(inputs, -100),
            "puzzle_identifiers": torch.zeros((1,), dtype=torch.int32, device=device),
        }

        carry = _prime_trm_carry(self.model, batch, z_H=z_H, z_L=z_L)
        last_outputs = None
        steps = 0
        cap = int(self.model.config.halt_max_steps)
        intermediates = [] if return_intermediates else None
        z_H_hist = [] if return_latents else None
        z_L_hist = [] if return_latents else None
        residuals = [] if track_res else None
        # Valid only when latents were injected; otherwise the first forward
        # resets them to H_init/L_init and there is no "previous" state to diff.
        prev_z_H = carry.inner_carry.z_H if not bool(carry.halted.any()) else None
        while True:
            carry, last_outputs = self.model(carry, batch)
            steps += 1
            if return_intermediates:
                intermediates.append(self.task.decode(last_outputs["logits"]))
            if return_latents:
                z_H_hist.append(carry.inner_carry.z_H.detach().float().cpu())
                z_L_hist.append(carry.inner_carry.z_L.detach().float().cpu())
            if track_res:
                residuals.append(
                    float("nan") if prev_z_H is None
                    else float(_rms_update(carry.inner_carry.z_H, prev_z_H)[0])
                )
                prev_z_H = carry.inner_carry.z_H
            if bool(carry.halted.all()) or steps >= cap:
                break

        extra = {"q_halt_logits": float(last_outputs["q_halt_logits"][0]),
                 "task": self.task.name}
        if return_latents:
            extra["z_H_history"] = torch.stack(z_H_hist).squeeze(1)   # (steps, seq, hidden)
            extra["z_L_history"] = torch.stack(z_L_hist).squeeze(1)

        grid = self.task.decode(last_outputs["logits"])
        return SolveResult(
            grid=grid,
            steps=steps,
            solved=self.task.is_solved(puzzle, grid),
            halted=bool(carry.halted.all()),
            solver=self.name,
            residual=residuals[-1] if residuals else None,   # last RMS latent update
            opt_time=None,
            intermediates=intermediates,
            residuals=residuals,
            extra=extra,
        )

    # ------------------------------------------------------------------ batch
    @torch.no_grad()
    def solve_batch(
        self,
        puzzle: Union[str, Grid],
        *,
        initial_latents: Optional["torch.Tensor"] = None,
        z_H: Optional["torch.Tensor"] = None,
        z_L: Optional["torch.Tensor"] = None,
        return_intermediates: bool = False,
        return_latents: bool = False,
        return_residuals: bool = False,
        max_steps: Optional[int] = None,
        **model_kwargs,
    ) -> BatchSolveResult:
        """Solve the same puzzle from B injected latent conditions in one batch.

        ``z_H`` / ``z_L`` are batched starting latents of shape ``(B, 916, 512)``
        (either may be given; the other keeps its trained default init).
        ``initial_latents`` is accepted as an alias for ``z_H`` -- the answer
        latent, TRM's closest analogue of FPRM's single latent -- to match the
        FPRM batched API.  There is no ``seeds`` mode: TRM's defaults are fixed
        buffers, so identical seeds would give identical rows.

        ``intermediates`` come back shaped ``[steps][B]`` so
        ``np.swapaxes(np.array(result.intermediates), 0, 1)`` gives
        ``(B, steps, 30, 30)`` like the other solvers.  With
        ``return_latents=True``, ``extra["z_H_history"]`` / ``z_L_history`` are
        ``(steps, B, 916, 512)`` float32 CPU tensors and ``residual_history``
        is ``[steps][B]`` RMS z_H updates.  ``return_residuals=True`` fills
        ``residual_history`` alone, without the latent histories (which cost
        ~3.8 MB per condition per step in float32) -- use this for large
        basin sweeps where only convergence times are needed.
        """
        if initial_latents is not None:
            if z_H is not None:
                raise ValueError("Pass either initial_latents (alias for z_H) or z_H, not both.")
            z_H = initial_latents
        if z_H is None and z_L is None:
            raise ValueError("Provide z_H/initial_latents and/or z_L of shape (B, 916, 512).")
        if max_steps is not None:
            self.model.config.halt_max_steps = int(max_steps)

        ref = torch.as_tensor(z_H if z_H is not None else z_L)
        B = ref.shape[0] if ref.ndim == 3 else 1

        device = next(self.model.parameters()).device
        inputs = self.task.encode(puzzle).to(device).repeat(B, 1)          # [B, seq_len]
        batch = {
            "inputs": inputs,
            "labels": torch.full_like(inputs, -100),
            "puzzle_identifiers": torch.zeros((B,), dtype=torch.int32, device=device),
        }

        carry = _prime_trm_carry(self.model, batch, z_H=z_H, z_L=z_L)
        last_outputs = None
        steps = 0
        cap = int(self.model.config.halt_max_steps)
        track_res = return_latents or return_residuals
        intermediates = [] if return_intermediates else None
        z_H_hist = [] if return_latents else None
        z_L_hist = [] if return_latents else None
        residual_history = [] if track_res else None
        prev_z_H = carry.inner_carry.z_H     # injected latents -> always a valid previous state
        while True:
            carry, last_outputs = self.model(carry, batch)
            steps += 1
            if return_intermediates:
                intermediates.append(self.task.decode_batch(last_outputs["logits"]))
            if return_latents:
                z_H_hist.append(carry.inner_carry.z_H.detach().float().cpu())
                z_L_hist.append(carry.inner_carry.z_L.detach().float().cpu())
            if track_res:
                residual_history.append(_rms_update(carry.inner_carry.z_H, prev_z_H).tolist())
                prev_z_H = carry.inner_carry.z_H
            if bool(carry.halted.all()) or steps >= cap:
                break

        extra = {"q_halt_logits": last_outputs["q_halt_logits"].float().cpu().tolist(),
                 "task": self.task.name}
        if return_latents:
            extra["z_H_history"] = torch.stack(z_H_hist)   # (steps, B, seq, hidden)
            extra["z_L_history"] = torch.stack(z_L_hist)

        grids = self.task.decode_batch(last_outputs["logits"])
        return BatchSolveResult(
            grids=grids,
            steps=steps,
            solved=[self.task.is_solved(puzzle, g) for g in grids],
            halted=bool(carry.halted.all()),
            solver=self.name,
            intermediates=intermediates,
            residual_history=residual_history,
            extra=extra,
        )


register_solver("trm", TRMSolver)
