"""High-level wrapper: load the FPRM checkpoint and solve Sudoku puzzles.

Encoding follows the HRM/TRM Sudoku dataset convention (which the checkpoint's
vocab size of 11 reflects):

    token = cell_digit + 1     (blank 0 -> 1, digits 1..9 -> 2..10)
    PAD id = 0  (unused for real cells)
    puzzle_identifier = 0

A puzzle is a flattened sequence of 81 cells.  The prediction is decoded back
with ``argmax(logits) - 1``.
"""
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import yaml
import torch

from .config import FPTRMConfig
from .model import FPTRM, Carry

# ----------------------------------------------------------------------- encoding

from ..puzzle import Grid, parse_puzzle, format_grid, is_valid_solution  # re-exported


def encode_puzzle(puzzle: Union[str, Grid]) -> torch.Tensor:
    """9x9 puzzle -> int32 input tokens of shape [1, 81] (digit + 1)."""
    grid = parse_puzzle(puzzle)
    flat = [grid[r][c] for r in range(9) for c in range(9)]
    tokens = [v + 1 for v in flat]
    return torch.tensor(tokens, dtype=torch.int32).unsqueeze(0)


def decode_solution(logits: torch.Tensor) -> Grid:
    """[1, 81, vocab] logits -> 9x9 grid (argmax - 1)."""
    pred = logits.argmax(dim=-1).squeeze(0).tolist()      # 81 tokens
    digits = [t - 1 for t in pred]
    return [digits[r * 9:(r + 1) * 9] for r in range(9)]


# ------------------------------------------------------------------------- loading

def _strip_prefix(state_dict, prefix="_orig_mod.model."):
    out = {}
    for k, v in state_dict.items():
        nk = k[len(prefix):] if k.startswith(prefix) else k
        out[nk] = v
    return out


# Arch deltas that turn the Sudoku FPRM config into the Maze-Hard FPRM config.
# Confirmed against the released maze checkpoint (loads with 0 missing/unexpected
# keys) and the model's own training ``all_config.yaml`` (data/maze-30x30-hard-1k).
# Only these differ from Sudoku; everything else (hidden 512, L_layers 2, num_heads
# 8, rope, puzzle_emb_len 16, alpha 0.75/0.25, ...) is shared:
#   * vocab_size 11 -> 6            (charset "# SGo" + pad)
#   * seq_len 81 -> 900             (30x30 board)
#   * conv 2D(3x3) -> causal 1D(k=4)
#   * norm_placement none -> output (RMS-norm the loop output; see transformer.py)
#   * L_cycles 6 -> 6 / max_iter(train) 24  (informational at eval)
#   * eval FP schedule: stepsize_decay 0.9 (as trained), fp_thresh 0.1
_FPRM_MAZE_ARCH_OVERRIDES = dict(
    vocab_size=6,
    seq_len=900,
    conv_type="conv1d",
    conv_kernel_size=4,
    norm_placement="output",
    L_cycles=6,
    max_iter=24,
    stepsize_decay=0.9,
    fp_thresh=0.1,
)


def _maze_arch() -> dict:
    """The FPRM Maze-Hard ``arch`` dict (Sudoku arch + the maze deltas)."""
    with open(os.path.join(os.path.dirname(__file__), "all_config.yaml")) as f:
        arch = dict(yaml.safe_load(f)["arch"])
    arch.update(_FPRM_MAZE_ARCH_OVERRIDES)
    return arch


def load_pretrained(
    checkpoint_path: str,
    config_path: Optional[str] = None,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
    arch_overrides: Optional[dict] = None,
) -> FPTRM:
    """Build the model from the yaml ``arch`` config and load the EMA checkpoint.

    The checkpoint stores parameters as ``_orig_mod.model.inner.*``; the
    ``_orig_mod.model.`` prefix is stripped to match ``FPTRM`` (which has an
    ``inner`` submodule).  Runs in float32 on CPU by default for determinism.

    ``config_path`` defaults to ``all_config.yaml`` next to the checkpoint; if that
    file is absent (e.g. the bare Maze checkpoint), pass ``config_path`` pointing at
    a yaml with an ``arch`` block, or supply ``arch_overrides`` to patch the Sudoku
    base arch (this is how the Maze solver is built -- see :func:`_maze_arch`).
    """
    if config_path is None:
        config_path = os.path.join(os.path.dirname(checkpoint_path), "all_config.yaml")

    if os.path.exists(config_path):
        with open(config_path) as f:
            arch = dict(yaml.safe_load(f)["arch"])
    else:
        # No sidecar config (bare checkpoint): start from the Sudoku base arch.
        with open(os.path.join(os.path.dirname(__file__), "all_config.yaml")) as f:
            arch = dict(yaml.safe_load(f)["arch"])
    if arch_overrides:
        arch.update(arch_overrides)
    # Use the requested dtype for forward computation (float32 by default).
    arch["forward_dtype"] = {torch.float32: "float32", torch.bfloat16: "bfloat16",
                             torch.float16: "float16"}.get(dtype, "float32")

    model = FPTRM(arch)

    sd = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if isinstance(sd, dict) and "model" in sd and "ema" in sd:
        # A full train_state.pt was passed: prefer the EMA weights.
        sd = sd["ema"]
    sd = _strip_prefix(sd)

    missing, unexpected = model.load_state_dict(sd, strict=False)
    # Buffers that are not in the checkpoint (rope caches) are expected to be missing.
    real_missing = [k for k in missing if not (k.endswith("cos_cached") or k.endswith("sin_cached"))]
    if real_missing or unexpected:
        raise RuntimeError(f"State dict mismatch.\n  missing: {real_missing}\n  unexpected: {unexpected}")

    model = model.to(device=device, dtype=dtype)
    model.eval()
    return model


# -------------------------------------------------------------------------- solving

@dataclass
class SolveInfo:
    steps: int            # number of fixed-point loop iterations taken
    residual: float       # final relative L_inf residual
    halted: bool          # whether the fixed-point halting condition fired
    # "Optimizer time" of the fixed-point iteration: the running sum of the
    # per-iteration step sizes  opt_time = sum_i dt_i,  where dt_i is the damping
    # factor eta used at iteration i (the update y <- y + eta*(f(y)-y) is an Euler
    # step of size eta).  With a fixed step this is just steps*eta; because FPOPT
    # decays eta, the meaningful "time" coordinate is this cumulative sum.
    opt_time: float = 0.0
    # Cumulative optimizer time after each iteration: cumulative_time[i] is
    # sum(dt_0..dt_i), so cumulative_time[-1] == opt_time.  Always populated.
    cumulative_time: Optional[List[float]] = None
    # Per-iteration step sizes dt_i (the eta applied at each iteration).  Always
    # populated; cumulative_time is its prefix-sum.
    stepsizes: Optional[List[float]] = None
    # Per-iteration decoded grids (intermediates[i] is the solution after loop
    # iteration i+1).  Populated only when solve_sudoku(..., return_intermediates=True);
    # otherwise None.  intermediates[-1] equals the returned grid.
    intermediates: Optional[List[Grid]] = None
    # Per-iteration residuals aligned with ``intermediates`` (same condition).
    residuals: Optional[List[float]] = None


def _prime_initial_latent(model, carry, batch, initial_latent):
    """Return a carry whose fixed-point latent z0 is ``initial_latent``.

    Normally the latent is (re)set to zeros inside ``reset_carry`` on the first
    forward pass (because every sample starts ``halted``).  To inject a custom z0
    we materialise the inner state once, overwrite its ``y``, copy the real puzzle
    into ``current_data``, and clear ``halted`` so the subsequent forward passes
    keep our latent instead of resetting it back to zeros.
    """
    device = next(model.parameters()).device
    # The injected latent state runs at the *parameter* dtype, not forward_dtype:
    # they agree for fp32/bf16/fp16 loads, but a float64-loaded model keeps
    # forward_dtype float32 (the input-injection constant stays fp32-valued, as
    # in the fractal-search branch), while the state itself must be fp64 or the
    # whole recurrence silently rounds to fp32.
    dtype = next(model.parameters()).dtype
    inner = model.inner
    B = batch["inputs"].shape[0]
    seq_len = batch["inputs"].shape[1]
    expected = (B, seq_len + inner.puzzle_emb_len, model.config.hidden_size)

    z = torch.as_tensor(initial_latent, device=device, dtype=dtype)
    if z.ndim == 2:
        z = z.unsqueeze(0)
    if tuple(z.shape) != expected:
        raise ValueError(
            f"initial_latent must have shape {expected} or {expected[1:]}, "
            f"got {tuple(initial_latent.shape)}"
        )

    # Materialise the inner optimizer state (y=zeros, eta=eta0, ...).
    inner_carry = inner.reset_carry(carry.halted, batch["inputs"], carry.inner_carry)
    inner_carry.z_L_state["y"] = z

    # Seed current_data with the actual puzzle (reset_carry doesn't touch it; the
    # wrapper's forward normally fills it on the halted reset, which we skip).
    current_data = {k: v.clone() for k, v in batch.items()}

    return Carry(
        inner_carry=inner_carry,
        steps=torch.zeros_like(carry.steps),
        halted=torch.zeros_like(carry.halted),   # not halted -> keep our latent
        current_data=current_data,
    )


@torch.no_grad()
def solve_sudoku(
    model: FPTRM,
    puzzle: Union[str, Grid],
    max_iters: Optional[int] = None,
    fp_thresh: Optional[float] = 0.02,
    return_intermediates: bool = False,
    initial_latent: Optional["torch.Tensor"] = None,
    stepsize: Optional[float] = None,
    stepsize_decay: Optional[float] = None,
    decay_patience: Optional[int] = None,
    task=None,
) -> Tuple[Grid, SolveInfo]:
    """Solve a Sudoku puzzle with the trained FPRM model.

    ``task`` selects the puzzle codec (default: the Sudoku task).  Pass the Maze
    task (``fprm.tasks.MAZE`` or ``"maze"``) with a maze-loaded model to solve
    mazes; the fixed-point machinery and the ``initial_latent`` API are identical.

    The model loops its fixed-point map until the relative residual drops below
    ``fp_thresh`` (the halting criterion) or ``max_iters`` is reached, spending
    more iterations on harder puzzles.

    ``fp_thresh`` defaults to 0.02 rather than the paper's nominal 0.1: empirically
    the residual can dip transiently below 0.1 on the hardest puzzles before the
    latent has truly settled, so a slightly tighter tolerance gives reliable exact
    solutions across difficulties.  Pass ``fp_thresh=0.1`` to reproduce the paper's
    nominal setting (faster, occasionally halts a step early on extreme puzzles).

    If ``return_intermediates`` is True, the model's decoded grid after *every*
    loop iteration is recorded in ``info.intermediates`` (with the matching
    per-iteration residuals in ``info.residuals``), letting you watch the puzzle
    fill in as the latent converges.  ``info.intermediates[-1]`` is the returned
    grid.

    ``initial_latent`` sets the starting latent state z0 of the fixed-point loop.
    By default (None) it is all zeros, matching the model's normal behaviour (the
    puzzle enters via the per-iteration input injection, not via z0).  Pass a
    tensor of shape ``(seq_len + puzzle_emb_len, hidden_size)`` or
    ``(1, seq_len + puzzle_emb_len, hidden_size)`` -- i.e. ``(97, 512)`` for the
    default Sudoku config -- to start the loop from a different initialisation.

    FPOPT step-size schedule (Algorithm 1) -- override the model's trained
    defaults to control how the damped iteration spends compute:

    * ``stepsize``       : initial damping eta_0 (default 1.0).  The update is
                           ``y <- eta * f(y) + (1 - eta) * y``; eta is the
                           "learning rate" of the fixed-point iteration.
    * ``stepsize_decay`` : geometric decay factor gamma in (0, 1] (default 0.9).
                           eta is multiplied by gamma when the residual plateaus.
                           Closer to 1.0 = less damping, iteration stays "hotter".
    * ``decay_patience`` : number of non-improving steps tolerated before eta is
                           decayed (default 5).

    These are written to ``model.config`` and read live by the optimizer, so they
    persist on ``model`` after the call (pass the value again, or rebuild/reload
    the model, to reset).

    Returns the predicted 9x9 grid and a :class:`SolveInfo` with adaptivity
    statistics (loop iterations, final residual).
    """
    from ..tasks import get_task, SUDOKU
    task = SUDOKU if task is None else get_task(task)

    device = next(model.parameters()).device
    inputs = task.encode(puzzle).to(device)                            # [1, seq_len]
    batch = {
        "inputs": inputs,
        "labels": torch.full_like(inputs, -100),                        # dummy (unused at inference)
        "puzzle_identifiers": torch.zeros((1,), dtype=torch.int32, device=device),
    }

    if max_iters is not None:
        model.config.max_iter_eval = max_iters
    if fp_thresh is not None:
        model.config.fp_thresh = fp_thresh
    if stepsize is not None:
        model.config.stepsize = stepsize
    if stepsize_decay is not None:
        model.config.stepsize_decay = stepsize_decay
    if decay_patience is not None:
        model.config.decay_patience = decay_patience

    carry = model.initial_carry(batch)
    if initial_latent is not None:
        carry = _prime_initial_latent(model, carry, batch, initial_latent)
    last_outputs = None
    steps = 0
    cap = model.max_iter
    intermediates: Optional[List[Grid]] = [] if return_intermediates else None
    residuals: Optional[List[float]] = [] if return_intermediates else None
    opt_time = 0.0
    cumulative_time: List[float] = []
    stepsizes: List[float] = []
    while True:
        carry, last_outputs = model(carry, batch)
        steps += 1
        # dt_i = the step size (damping eta) actually applied this iteration.
        dt = float(carry.inner_carry.z_L_state["dt"].max().item())
        opt_time += dt                            # running sum over dt_i
        stepsizes.append(dt)
        cumulative_time.append(opt_time)
        if return_intermediates:
            intermediates.append(task.decode(last_outputs["logits"]))
            residuals.append(float(carry.inner_carry.z_L_state["residues"].max().item()))
        if bool(carry.halted.all()) or steps >= cap:
            break

    grid = task.decode(last_outputs["logits"])
    info = SolveInfo(
        steps=steps,
        residual=float(carry.inner_carry.z_L_state["residues"].max().item()),
        halted=bool(carry.halted.all()),
        opt_time=opt_time,
        cumulative_time=cumulative_time,
        stepsizes=stepsizes,
        intermediates=intermediates,
        residuals=residuals,
    )
    return grid, info


# --------------------------------------------------------------- batched solving

def _decode_batch(logits: torch.Tensor) -> List[Grid]:
    """[B, 81, vocab] logits -> list of B 9x9 grids (argmax - 1)."""
    preds = (logits.argmax(dim=-1) - 1).tolist()           # [B, 81]
    return [[row[r * 9:(r + 1) * 9] for r in range(9)] for row in preds]


@dataclass
class BatchSolveInfo:
    """Per-sample diagnostics for a batched solve (B samples solved together)."""
    steps: int                       # loop iterations (same for all samples; see note)
    residuals: List[float]           # final per-sample residual, length B
    halted: bool                     # whether the global halting condition fired
    opt_time: float                  # cumulative sum of dt_i (shared step schedule)
    cumulative_time: List[float]     # prefix sum of dt over iterations, length `steps`
    stepsizes: List[float]           # dt_i per iteration, length `steps`
    # Per-iteration trajectories (always populated):
    residual_history: List[List[float]]   # [steps][B] relative residual per sample
    # Per-iteration decoded grids, only if return_intermediates=True:
    intermediates: Optional[List[List[Grid]]] = None   # [steps] -> list of B grids


@torch.no_grad()
def solve_sudoku_batch(
    model: FPTRM,
    puzzle: Union[str, Grid],
    initial_latents: "torch.Tensor",
    max_iters: Optional[int] = None,
    fp_thresh: Optional[float] = 0.02,
    return_intermediates: bool = False,
    stepsize: Optional[float] = None,
    stepsize_decay: Optional[float] = None,
    decay_patience: Optional[int] = None,
    task=None,
) -> Tuple[List[Grid], BatchSolveInfo]:
    """Solve the *same* puzzle from B different initial latents, in one batched pass.

    ``task`` selects the puzzle codec (default: Sudoku).  The initial-latent
    batching is identical across tasks -- only the latent shape's ``seq_len``
    differs (Sudoku ``(B, 97, 512)``; Maze ``(B, 916, 512)``), which is validated
    against the model's own config, so the same call works for either.

    This is the efficient way to study how the solution path varies with the
    initial condition (e.g. across seeds): the model is already batched over its
    leading dimension, so B trajectories run as a single ``B x 97 x 512`` forward
    per iteration instead of B separate solves -- far cheaper than multiprocessing,
    which would copy the model per worker and serialise results.

    Parameters
    ----------
    initial_latents : tensor of shape ``(B, seq_len + puzzle_emb_len, hidden_size)``
                      = ``(B, 97, 512)`` for the default Sudoku config.  B is taken
                      from this tensor; the puzzle is repeated across the batch.

    Returns
    -------
    (grids, info) where ``grids`` is a list of B final 9x9 grids and ``info`` is a
    :class:`BatchSolveInfo` with per-sample residual trajectories.

    Note on halting: FPRM halts the *whole batch* together (the loop stops when the
    slowest sample's residual drops below ``fp_thresh``), so ``steps`` is shared.
    Per-sample convergence is recoverable from ``info.residual_history`` -- find,
    per column, the first iteration where the residual falls below ``fp_thresh``.
    """
    from ..tasks import get_task, SUDOKU
    task = SUDOKU if task is None else get_task(task)

    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype   # see _prime_initial_latent: fp64-safe

    z0 = torch.as_tensor(initial_latents, device=device, dtype=dtype)
    if z0.ndim != 3:
        raise ValueError(
            f"initial_latents must be 3-D (B, seq_len+puzzle_emb_len, hidden_size); "
            f"got shape {tuple(z0.shape)}"
        )
    B = z0.shape[0]

    inputs = task.encode(puzzle).to(device).repeat(B, 1)               # [B, seq_len]
    batch = {
        "inputs": inputs,
        "labels": torch.full_like(inputs, -100),
        "puzzle_identifiers": torch.zeros((B,), dtype=torch.int32, device=device),
    }

    if max_iters is not None:
        model.config.max_iter_eval = max_iters
    if fp_thresh is not None:
        model.config.fp_thresh = fp_thresh
    if stepsize is not None:
        model.config.stepsize = stepsize
    if stepsize_decay is not None:
        model.config.stepsize_decay = stepsize_decay
    if decay_patience is not None:
        model.config.decay_patience = decay_patience

    carry = model.initial_carry(batch)
    carry = _prime_initial_latent(model, carry, batch, z0)

    last_outputs = None
    steps = 0
    cap = model.max_iter
    opt_time = 0.0
    cumulative_time: List[float] = []
    stepsizes: List[float] = []
    residual_history: List[List[float]] = []
    intermediates: Optional[List[List[Grid]]] = [] if return_intermediates else None
    while True:
        carry, last_outputs = model(carry, batch)
        steps += 1
        z_state = carry.inner_carry.z_L_state
        dt = float(z_state["dt"].max().item())     # shared step schedule
        opt_time += dt
        stepsizes.append(dt)
        cumulative_time.append(opt_time)
        residual_history.append(z_state["residues"].tolist())          # [B]
        if return_intermediates:
            intermediates.append(task.decode_batch(last_outputs["logits"]))
        if bool(carry.halted.all()) or steps >= cap:
            break

    grids = task.decode_batch(last_outputs["logits"])
    info = BatchSolveInfo(
        steps=steps,
        residuals=carry.inner_carry.z_L_state["residues"].tolist(),
        halted=bool(carry.halted.all()),
        opt_time=opt_time,
        cumulative_time=cumulative_time,
        stepsizes=stepsizes,
        residual_history=residual_history,
        intermediates=intermediates,
    )
    return grids, info
