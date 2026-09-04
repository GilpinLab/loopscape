"""FPRM solver -- adapter over the existing fprm.solver code.

Wraps ``load_pretrained`` + ``solve_sudoku`` so FPRM is usable through the unified
:class:`Solver` interface.  The underlying functions are unchanged, so existing
callers of ``solve_sudoku`` keep working.

Two tasks are supported (``task="sudoku"`` default, or ``"maze"``).  The maze model
is the same architecture with a causal 1D short-conv (k=4), ``norm_placement=
"output"`` and a maze-sized vocab/seq_len; the ``solve`` / ``solve_batch`` API --
including the batched ``initial_latents`` sweep -- is identical across tasks.
"""
from __future__ import annotations

import os
from typing import Optional, Union

import torch

from .inference import Grid, load_pretrained, solve_sudoku, solve_sudoku_batch, _maze_arch
from ..tasks import get_task
from ..base import BatchSolveResult, Solver, SolveResult, register_solver

from ..paths import FPRM_WEIGHTS_DIR
from .download import ensure_fprm_checkpoint

# Default (final) checkpoints, cached under the shared weights dir and
# downloaded from hf.co/fixed-point-reasoners/fprm on first use.
_DEFAULT_CKPT = {
    "sudoku": os.path.join(str(FPRM_WEIGHTS_DIR), "sudoku", "step_78120"),
    "maze": os.path.join(str(FPRM_WEIGHTS_DIR), "maze", "step_78120"),
}


class FPRMSolver(Solver):
    """Fixed-Point Reasoning Model -- deterministic, residual-halting solver."""

    name = "fprm"

    def __init__(self, model, device: str = "cpu", task="sudoku"):
        super().__init__(model, device=device)
        self.task = get_task(task)

    @classmethod
    def load(
        cls,
        checkpoint: Optional[str] = None,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
        task: str = "sudoku",
        **kwargs,
    ) -> "FPRMSolver":
        t = get_task(task)
        ckpt = checkpoint or _DEFAULT_CKPT[t.name]
        if checkpoint is None and not os.path.exists(ckpt):
            ckpt = ensure_fprm_checkpoint(t.name)   # fetch the released final weights
        # The maze checkpoint ships without an all_config.yaml sidecar; build its
        # arch from the confirmed maze deltas (see fprm.solver._maze_arch).
        overrides = _maze_arch() if t.name == "maze" else None
        model = load_pretrained(ckpt, device=device, dtype=dtype,
                                arch_overrides=overrides, **kwargs)
        return cls(model, device=device, task=t)

    def solve(
        self,
        puzzle: Union[str, Grid],
        *,
        return_intermediates: bool = False,
        **model_kwargs,
    ) -> SolveResult:
        """Solve a puzzle with FPRM.

        Accepts the same tuning knobs as ``solve_sudoku`` via ``model_kwargs``
        (``fp_thresh``, ``max_iters``, ``stepsize``, ``stepsize_decay``,
        ``decay_patience``, ``initial_latent``).
        """
        grid, info = solve_sudoku(
            self.model, puzzle, return_intermediates=return_intermediates,
            task=self.task, **model_kwargs
        )
        return SolveResult(
            grid=grid,
            steps=info.steps,
            solved=self.task.is_solved(puzzle, grid),
            halted=info.halted,
            solver=self.name,
            residual=info.residual,
            opt_time=info.opt_time,
            intermediates=info.intermediates,
            residuals=info.residuals,
            extra={"stepsizes": info.stepsizes, "cumulative_time": info.cumulative_time,
                   "task": self.task.name},
        )

    def solve_batch(
        self,
        puzzle: Union[str, Grid],
        *,
        initial_latents: "torch.Tensor",
        return_intermediates: bool = False,
        **model_kwargs,
    ) -> BatchSolveResult:
        """Solve the puzzle from B initial latents in one batched pass.

        ``initial_latents`` is a tensor of shape ``(B, seq_len + puzzle_emb_len,
        hidden_size)`` -- ``(B, 97, 512)`` for Sudoku, ``(B, 916, 512)`` for Maze
        (one starting latent per condition).  Extra knobs (``fp_thresh``,
        ``max_iters``, ``stepsize``, ...) pass through to ``solve_sudoku_batch``.
        """
        grids, info = solve_sudoku_batch(
            self.model, puzzle, initial_latents,
            return_intermediates=return_intermediates, task=self.task, **model_kwargs,
        )
        return BatchSolveResult(
            grids=grids,
            steps=info.steps,
            solved=[self.task.is_solved(puzzle, g) for g in grids],
            halted=info.halted,
            solver=self.name,
            intermediates=info.intermediates,
            residual_history=info.residual_history,
            extra={
                "residuals": info.residuals,
                "stepsizes": info.stepsizes,
                "cumulative_time": info.cumulative_time,
                "opt_time": info.opt_time,
                "task": self.task.name,
            },
        )


register_solver("fprm", FPRMSolver)
