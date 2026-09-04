"""EqR solver -- Equilibrium Reasoners (arXiv:2605.21488).

EqR is a hierarchical (z_H/z_L), HRM-style recursive reasoner with **stochastic**
inference (Gaussian noise injected each step, latents randomly initialised) and
ACT-style halting (a fixed ``halt_max_steps`` at eval).  This contrasts with
FPRM's deterministic fixed-point loop.

The model code is a PINNED upstream dependency (github.com/locuslab/EqR at the
commit recorded in :mod:`loopscape.eqr.download`), fetched on first use and
imported from ``sys.path`` -- nothing is vendored.  Weights are downloaded from
the Hugging Face Hub on first use (same module).
"""
from __future__ import annotations

import os
import sys
from typing import Optional, Union

import torch

from ..puzzle import Grid
from ..tasks import get_task
from ..base import BatchSolveResult, Solver, SolveResult, register_solver
from .download import ensure_eqr_checkpoint, ensure_eqr_source

def _import_eqr_model():
    """Import ``EqRModel`` from the pinned upstream EqR source.

    The upstream codebase (github.com/locuslab/EqR, pinned by commit in
    :mod:`loopscape.eqr.download`) is downloaded on first use and added to
    ``sys.path`` -- EqR uses absolute imports (``from models... import ...``).
    Only the model + its layer/embedding modules are needed for inference; no
    hydra/omegaconf-driven training code is touched.  Set ``LOOPSCAPE_EQR_SRC``
    to point at a local EqR checkout instead (e.g. a modified training copy).
    """
    eqr_dir = ensure_eqr_source()
    if eqr_dir not in sys.path:
        sys.path.insert(0, eqr_dir)

    # ``models.layers`` imports ``utils.printing.colored_exception``, which pulls
    # in ``colorama`` and is only used on a CUDA-only error path.  Stub it so the
    # model imports without that extra dependency (we run CPU/SDPA).
    if "utils.printing" not in sys.modules:
        import types

        def _colored_exception(exc_type, message):  # matches the upstream signature
            raise exc_type(message)

        stub = types.ModuleType("utils.printing")
        for _name in ("colored_exception", "rank_zero_print_info",
                      "rank_zero_print_warning", "rank_zero_print"):
            setattr(stub, _name, _colored_exception if _name == "colored_exception"
                    else (lambda *a, **k: None))
        pkg = sys.modules.get("utils") or types.ModuleType("utils")
        pkg.printing = stub
        sys.modules.setdefault("utils", pkg)
        sys.modules["utils.printing"] = stub

    from models.eqr import EqRModel  # noqa: E402  (path set up above)

    # EqR's Attention hard-requires flash_attn whenever tensors are on CUDA and
    # raises if it isn't installed.  Shim it with torch's built-in SDPA (same
    # math; fused kernels on GPU): flash_attn uses (B, N, heads, dim) layout,
    # SDPA wants (B, heads, N, dim), hence the permutes -- mirroring the
    # module's own CPU branch.
    import models.layers as _eqr_layers  # noqa: E402
    if _eqr_layers.flash_attn_func is None:
        import torch.nn.functional as _E

        def _sdpa_flash_attn(q, k, v, causal=False):
            return _E.scaled_dot_product_attention(
                q.permute(0, 2, 1, 3), k.permute(0, 2, 1, 3), v.permute(0, 2, 1, 3),
                is_causal=causal,
            ).permute(0, 2, 1, 3)

        _eqr_layers.flash_attn_func = _sdpa_flash_attn

    return EqRModel


# Architecture config for the Sudoku EqR checkpoint.  Values match the config
# embedded in locuslab/EqR-model (arch/eqr.yaml + eqr_sudoku's mlp_t override),
# plus the dataset-derived fields (vocab 11, seq_len 81, single puzzle id).
_EQR_SUDOKU_CONFIG = dict(
    batch_size=1,
    seq_len=81,
    vocab_size=11,
    num_puzzle_identifiers=1,
    H_cycles=3,
    L_cycles=6,
    H_layers=0,
    L_layers=2,
    hidden_size=512,
    num_heads=8,
    expansion=4,
    pos_encodings="none",
    board_height=9,
    board_width=9,
    halt_max_steps=16,
    halt_exploration_prob=0.1,
    forward_dtype="bfloat16",
    mlp_t=True,
    puzzle_emb_len=16,
    puzzle_emb_ndim=512,
    lambda_=0.95,
    noise_scale=0.01,
)

# Architecture config for the Maze EqR checkpoint (locuslab/EqR-model,
# maze-unique/eqr.pth).  Values are the config embedded in that checkpoint:
# vocab 6 ("# SGo" + pad), seq_len 900 (30x30), a smaller/shallower recurrent core
# (hidden 128, L_layers 1, L_cycles 4) with rope attention (mlp_t off).
_EQR_MAZE_CONFIG = dict(
    batch_size=1,
    seq_len=900,
    vocab_size=6,
    num_puzzle_identifiers=1,
    H_cycles=3,
    L_cycles=4,
    H_layers=0,
    L_layers=1,
    hidden_size=128,
    num_heads=8,
    expansion=4,
    pos_encodings="rope",
    board_height=30,
    board_width=30,
    halt_max_steps=16,
    halt_exploration_prob=0.1,
    forward_dtype="bfloat16",
    mlp_t=False,
    puzzle_emb_len=16,
    puzzle_emb_ndim=128,
    lambda_=0.95,
    noise_scale=0.01,
)

_EQR_CONFIG = {"sudoku": _EQR_SUDOKU_CONFIG, "maze": _EQR_MAZE_CONFIG}

_PREFIX = "_orig_mod.model."   # checkpoint keys -> EqRModel (which owns `.inner`)


def _rms_update(z_new: torch.Tensor, z_old: torch.Tensor) -> torch.Tensor:
    """Per-sample RMS latent update ||z_new - z_old|| -- EqR's residual analogue."""
    return (z_new.float() - z_old.float()).pow(2).mean(dim=(1, 2)).sqrt()


def _extract_state_dict(ckpt: dict) -> dict:
    """Pull a clean, prefix-stripped state_dict out of an EqR training checkpoint.

    Prefers EMA weights (merged over the raw model to fill any EMA-untracked
    buffers such as the sparse puzzle embedding).
    """
    model_sd = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    merged = dict(model_sd)
    ema = ckpt.get("ema") if isinstance(ckpt, dict) else None
    if isinstance(ema, dict) and isinstance(ema.get("shadow"), dict):
        merged.update(ema["shadow"])  # EMA weights take precedence

    out = {}
    for k, v in merged.items():
        out[k[len(_PREFIX):] if k.startswith(_PREFIX) else k] = v
    return out


def _latent_shape(model, batch) -> tuple:
    B = batch["inputs"].shape[0]
    return (B, model.config.seq_len, model.config.hidden_size)   # (B, 97, 512)


def _prepare_latent(z, model, batch, name: str):
    """Validate/broadcast a user-supplied z_H or z_L to (B, seq_len, hidden)."""
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


def _prime_eqr_carry(model, batch, z_H=None, z_L=None):
    """Build a carry that starts from user-supplied z_H/z_L instead of noise.

    Normally EqR's ``forward`` overwrites both latents with truncated-normal noise
    on the first step (every sample starts ``halted``).  Setting the latents and
    clearing ``halted`` makes the model keep them; ``current_data`` is seeded with
    the real puzzle since we skip the halted reset that normally fills it.
    Latents left as ``None`` keep EqR's default random initialisation.
    """
    carry = model.initial_carry(batch)
    if z_H is None and z_L is None:
        return carry

    # If only one latent is injected, initialise the other with EqR's own random
    # reset so it isn't left as uninitialised empty memory.
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


class EqRSolver(Solver):
    """Equilibrium Reasoner -- hierarchical, stochastic, ACT-halting solver."""

    name = "eqr"

    def __init__(self, model, device: str = "cpu", task="sudoku"):
        super().__init__(model, device=device)
        self.task = get_task(task)

    @classmethod
    def load(
        cls,
        checkpoint: Optional[str] = None,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
        config_overrides: Optional[dict] = None,
        task: str = "sudoku",
        **kwargs,
    ) -> "EqRSolver":
        EqRModel = _import_eqr_model()
        t = get_task(task)

        cfg = dict(_EQR_CONFIG[t.name])
        cfg["forward_dtype"] = {torch.float32: "float32", torch.bfloat16: "bfloat16",
                                torch.float16: "float16"}.get(dtype, "float32")
        if config_overrides:
            cfg.update(config_overrides)

        model = EqRModel(cfg)

        ckpt_path = ensure_eqr_checkpoint(checkpoint, task=t.name)
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state_dict = _extract_state_dict(ckpt)

        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        # Non-persistent buffers (rope caches, latent-init helpers) may be absent.
        real_missing = [k for k in missing
                        if not any(s in k for s in ("cos_cached", "sin_cached",
                                                    "H_init", "L_init", "_random_reset"))]
        if real_missing or unexpected:
            raise RuntimeError(
                f"EqR state dict mismatch.\n  missing: {real_missing}\n  unexpected: {unexpected}"
            )

        model = model.to(device=device, dtype=dtype)
        model.eval()
        return cls(model, device=device, task=t)

    @torch.no_grad()
    def solve(
        self,
        puzzle: Union[str, Grid],
        *,
        return_intermediates: bool = False,
        return_latents: bool = False,
        seed: Optional[int] = 0,
        max_steps: Optional[int] = None,
        noise_scale: Optional[float] = None,
        z_H: Optional["torch.Tensor"] = None,
        z_L: Optional["torch.Tensor"] = None,
        **model_kwargs,
    ) -> SolveResult:
        """Solve a puzzle with EqR.

        EqR is **stochastic** (noise injection + random latent init).  The RNG
        ``seed`` is fixed (default 0) for reproducibility; override it to vary the
        run, or pass ``None`` to leave the global RNG untouched.

        The initial latent state can be picked directly instead of randomised:

        Parameters
        ----------
        z_H, z_L    : optional starting latents, each of shape
                      ``(seq_len, hidden)`` = ``(97, 512)`` (or batched
                      ``(1, 97, 512)``).  Either may be given independently; the
                      other keeps EqR's random init.  Note EqR still injects
                      per-step noise (``noise_scale``), so the run is not a pure
                      function of the latents unless you also set ``noise_scale=0``.
        seed        : seed torch's RNG before solving (default 0; None = leave RNG).
        max_steps   : override ``halt_max_steps`` (the fixed eval loop length).
        noise_scale : override the per-step Gaussian noise magnitude.
        return_latents : record the per-step latent states.  The traces land in
                      ``extra["z_H_history"]`` / ``extra["z_L_history"]`` as
                      float32 CPU tensors of shape ``(steps, seq_len, hidden)``.
                      Also fills ``residuals`` with the per-step RMS latent
                      update ``||z_H(t) - z_H(t-1)||`` (EqR's residual
                      analogue; step 1 is NaN when the initial latent was
                      random, i.e. no z_H/z_L was injected).
        """
        if seed is not None:
            torch.manual_seed(seed)
        if max_steps is not None:
            self.model.config.halt_max_steps = int(max_steps)
        if noise_scale is not None:
            self.model.config.noise_scale = float(noise_scale)
            self.model.inner.L_level.noise_scale = float(noise_scale)

        device = next(self.model.parameters()).device
        inputs = self.task.encode(puzzle).to(device)                       # [1, seq_len]
        batch = {
            "inputs": inputs,
            "labels": torch.full_like(inputs, -100),
            "puzzle_identifiers": torch.zeros((1,), dtype=torch.int32, device=device),
        }

        carry = _prime_eqr_carry(self.model, batch, z_H=z_H, z_L=z_L)
        last_outputs = None
        steps = 0
        cap = int(self.model.config.halt_max_steps)
        intermediates = [] if return_intermediates else None
        z_H_hist = [] if return_latents else None
        z_L_hist = [] if return_latents else None
        residuals = [] if return_latents else None
        # Valid only when latents were injected; otherwise EqR randomises them
        # inside the first forward and there is no "previous" state to diff.
        prev_z_H = carry.inner_carry.z_H if not bool(carry.halted.any()) else None
        while True:
            carry, last_outputs = self.model(carry, batch)
            steps += 1
            if return_intermediates:
                intermediates.append(self.task.decode(last_outputs["logits"]))
            if return_latents:
                z_H_hist.append(carry.inner_carry.z_H.detach().float().cpu())
                z_L_hist.append(carry.inner_carry.z_L.detach().float().cpu())
                residuals.append(
                    float("nan") if prev_z_H is None
                    else float(_rms_update(carry.inner_carry.z_H, prev_z_H)[0])
                )
                prev_z_H = carry.inner_carry.z_H
            if bool(carry.halted.all()) or steps >= cap:
                break

        extra = {"seed": seed, "noise_scale": float(self.model.config.noise_scale)}
        if return_latents:
            extra["z_H_history"] = torch.cat(z_H_hist)     # (steps, seq, hidden)
            extra["z_L_history"] = torch.cat(z_L_hist)

        grid = self.task.decode(last_outputs["logits"])
        return SolveResult(
            grid=grid,
            steps=steps,
            solved=self.task.is_solved(puzzle, grid),
            halted=bool(carry.halted.all()),
            solver=self.name,
            residual=residuals[-1] if residuals else None,  # last RMS latent update
            opt_time=None,
            intermediates=intermediates,
            residuals=residuals,
            extra=extra,
        )

    @torch.no_grad()
    def solve_batch(
        self,
        puzzle: Union[str, Grid],
        *,
        seeds=None,
        z_H: Optional["torch.Tensor"] = None,
        z_L: Optional["torch.Tensor"] = None,
        return_intermediates: bool = False,
        return_latents: bool = False,
        seed: Optional[int] = 0,
        max_steps: Optional[int] = None,
        noise_scale: Optional[float] = None,
        **model_kwargs,
    ) -> BatchSolveResult:
        """Solve the same puzzle from B different initial conditions.

        Two ways to specify the B conditions (choose one):

        * ``z_H`` / ``z_L`` : batched starting latents of shape ``(B, 97, 512)``
          (either may be given; the other keeps EqR's random init).  All B run in a
          **single batched forward** -- the efficient way to sweep the latent state.
        * ``seeds``         : a list of B RNG seeds.  Each is run as its own
          reproducible trajectory (looped, since EqR's noise is global).

        The RNG ``seed`` (default 0) fixes the per-step noise for the latent-sweep
        mode; override or set ``noise_scale=0`` for a deterministic sweep.

        Returns a :class:`BatchSolveResult` with ``intermediates`` shaped
        ``[steps][B]`` so ``np.swapaxes(np.array(result.intermediates), 0, 1)`` gives
        ``(B, steps, 9, 9)`` -- matching the FPRM batched layout.

        With ``return_latents=True`` the per-step latent states are recorded:
        ``extra["z_H_history"]`` / ``extra["z_L_history"]`` are float32 CPU
        tensors of shape ``(steps, B, seq_len, hidden)`` (~5 MB per sample per
        24-step Sudoku trace, each).  ``residual_history`` is then filled with
        the per-step RMS latent update ``||z_H(t) - z_H(t-1)||`` shaped
        ``[steps][B]`` -- EqR's analogue of FPRM's fixed-point residual (step 1
        is NaN when the corresponding initial latent was random).
        """
        if (z_H is not None or z_L is not None) and seeds is not None:
            raise ValueError("Pass either z_H/z_L (latent sweep) or seeds, not both.")

        # -- Seed-sweep mode: loop named seeds (each a reproducible trajectory). --
        if z_H is None and z_L is None:
            if seeds is None:
                raise ValueError("Provide seeds=[...] or z_H/z_L=(B,97,512) to solve_batch.")
            seeds = list(seeds)
            results: List[SolveResult] = [
                self.solve(puzzle, return_intermediates=return_intermediates,
                           return_latents=return_latents, seed=s,
                           max_steps=max_steps, noise_scale=noise_scale, **model_kwargs)
                for s in seeds
            ]
            steps = max((r.steps for r in results), default=0)
            intermediates = None
            if return_intermediates:
                def at(res: SolveResult, i: int) -> Grid:
                    seq = res.intermediates or [res.grid]
                    return seq[i] if i < len(seq) else seq[-1]
                intermediates = [[at(r, i) for r in results] for i in range(steps)]
            extra = {"seeds": seeds, "noise_scale": float(self.model.config.noise_scale)}
            residual_history = None
            if return_latents:
                # Per-seed runs may halt at different steps; keep the traces as
                # per-seed lists rather than padding into one tensor.
                extra["z_H_history"] = [r.extra["z_H_history"] for r in results]
                extra["z_L_history"] = [r.extra["z_L_history"] for r in results]
                def res_at(res: SolveResult, i: int) -> float:
                    return res.residuals[i] if i < len(res.residuals) else res.residuals[-1]
                residual_history = [[res_at(r, i) for r in results] for i in range(steps)]
            return BatchSolveResult(
                grids=[r.grid for r in results],
                steps=steps,
                solved=[r.solved for r in results],
                halted=all(r.halted for r in results),
                solver=self.name,
                intermediates=intermediates,
                residual_history=residual_history,
                extra=extra,
            )

        # -- Latent-sweep mode: one batched forward over B injected latents. --
        if seed is not None:
            torch.manual_seed(seed)
        if max_steps is not None:
            self.model.config.halt_max_steps = int(max_steps)
        if noise_scale is not None:
            self.model.config.noise_scale = float(noise_scale)
            self.model.inner.L_level.noise_scale = float(noise_scale)

        # B is taken from whichever latent(s) were provided.
        ref = z_H if z_H is not None else z_L
        ref = torch.as_tensor(ref)
        B = ref.shape[0] if ref.ndim == 3 else 1

        device = next(self.model.parameters()).device
        inputs = self.task.encode(puzzle).to(device).repeat(B, 1)          # [B, seq_len]
        batch = {
            "inputs": inputs,
            "labels": torch.full_like(inputs, -100),
            "puzzle_identifiers": torch.zeros((B,), dtype=torch.int32, device=device),
        }

        carry = _prime_eqr_carry(self.model, batch, z_H=z_H, z_L=z_L)
        last_outputs = None
        steps = 0
        cap = int(self.model.config.halt_max_steps)
        intermediates = [] if return_intermediates else None
        z_H_hist = [] if return_latents else None
        z_L_hist = [] if return_latents else None
        residual_history = [] if return_latents else None
        prev_z_H = carry.inner_carry.z_H     # injected latents -> always a valid previous state
        while True:
            carry, last_outputs = self.model(carry, batch)
            steps += 1
            if return_intermediates:
                intermediates.append(self.task.decode_batch(last_outputs["logits"]))
            if return_latents:
                z_H_hist.append(carry.inner_carry.z_H.detach().float().cpu())
                z_L_hist.append(carry.inner_carry.z_L.detach().float().cpu())
                residual_history.append(_rms_update(carry.inner_carry.z_H, prev_z_H).tolist())
                prev_z_H = carry.inner_carry.z_H
            if bool(carry.halted.all()) or steps >= cap:
                break

        extra = {"noise_scale": float(self.model.config.noise_scale)}
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


register_solver("eqr", EqRSolver)
