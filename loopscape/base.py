"""Unified solver interface for the Sudoku reasoning models in this repo.

A :class:`Solver` wraps a trained model behind a common ``solve(puzzle)`` API so
callers can switch between models (FPRM, EqR, ...) without knowing their internals.
Use :func:`get_solver` to construct one by name.

Shared puzzle encoding/decoding/validation lives in :mod:`fprm.solver` and is
reused here -- both models use the same Sudoku tokenisation.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

from .puzzle import Grid, is_valid_solution, parse_puzzle


@dataclass
class SolveResult:
    """Model-agnostic result of a single solve.

    Fields common to all solvers are always populated; model-specific quantities
    are optional (``None`` when the model doesn't produce them) or live in
    ``extra``.
    """
    grid: Grid                         # decoded 9x9 solution
    steps: int                         # number of reasoning iterations taken
    solved: bool                       # givens preserved AND a valid completed grid
    halted: bool                       # whether the model's halting condition fired
    solver: str = ""                   # which solver produced this (e.g. "fprm")

    # FPRM-style fixed-point diagnostics (None for models without them, e.g. EqR):
    residual: Optional[float] = None
    opt_time: Optional[float] = None

    # Per-iteration trajectories (populated only when return_intermediates=True):
    intermediates: Optional[List[Grid]] = None
    residuals: Optional[List[float]] = None

    # Model-specific spillover (e.g. EqR: seed, noise_scale).
    extra: Dict[str, object] = field(default_factory=dict)


@dataclass
class BatchSolveResult:
    """Result of solving one puzzle from B different initial conditions.

    The intermediates layout matches the FPRM batched API so
    ``np.swapaxes(np.array(result.intermediates), 0, 1)`` yields ``(B, steps, 9, 9)``
    for either solver.
    """
    grids: List[Grid]                  # B final grids
    steps: int                         # iterations run (shared across the batch)
    solved: List[bool]                 # per-condition correctness, length B
    halted: bool                       # whether the (global) halting condition fired
    solver: str = ""

    # Per-iteration decoded grids: intermediates[i] is the list of B grids after
    # iteration i+1.  Populated only when return_intermediates=True.
    intermediates: Optional[List[List[Grid]]] = None      # [steps][B]
    # Per-iteration residuals: residual_history[i] is length B.  FPRM: the
    # fixed-point residual.  EqR (with return_latents=True): the RMS latent
    # update ||z_H(t) - z_H(t-1)||, its closest analogue.
    residual_history: Optional[List[List[float]]] = None  # [steps][B]

    extra: Dict[str, object] = field(default_factory=dict)


def _givens_preserved(puzzle: Union[str, Grid], solution: Grid) -> bool:
    g = parse_puzzle(puzzle)
    return all(g[r][c] == 0 or g[r][c] == solution[r][c] for r in range(9) for c in range(9))


def compute_solved(puzzle: Union[str, Grid], solution: Grid) -> bool:
    """A solve is correct iff every given is kept and the grid is a valid Sudoku."""
    return _givens_preserved(puzzle, solution) and is_valid_solution(solution)


class Solver(ABC):
    """Base class for a Sudoku solver backed by a trained model."""

    name: str = "solver"

    def __init__(self, model, device: str = "cpu"):
        self.model = model
        self.device = device

    @classmethod
    @abstractmethod
    def load(cls, **kwargs) -> "Solver":
        """Build the model, load its weights, and return a ready solver."""

    @abstractmethod
    def solve(self, puzzle: Union[str, Grid], *, return_intermediates: bool = False,
              **model_kwargs) -> SolveResult:
        """Solve one puzzle and return a :class:`SolveResult`."""

    def solve_batch(self, puzzle: Union[str, Grid], *, return_intermediates: bool = False,
                    **model_kwargs) -> "BatchSolveResult":
        """Solve the same puzzle from B different initial conditions in one batch.

        What "initial condition" means is model-specific and passed via
        ``model_kwargs``:

        * FPRM  -> ``initial_latents`` : tensor ``(B, 97, 512)`` of starting latents.
        * EqR   -> ``seeds``           : list of B RNG seeds (EqR is stochastic).

        Either way the result's ``intermediates`` are shaped ``[steps][B]`` so
        ``np.swapaxes(np.array(result.intermediates), 0, 1)`` gives ``(B, steps, 9, 9)``
        for any solver.
        """
        raise NotImplementedError(f"{type(self).__name__} does not implement solve_batch()")

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"{type(self).__name__}(name={self.name!r}, device={self.device!r})"


# Registry of available solvers, populated by the concrete modules.
_SOLVERS: Dict[str, type] = {}


def register_solver(name: str, cls: type) -> None:
    _SOLVERS[name.lower()] = cls


def available_solvers() -> List[str]:
    return sorted(_SOLVERS)


def get_solver(name: str, **load_kwargs) -> Solver:
    """Construct and load a solver by name.

    Parameters
    ----------
    name : "fprm" | "eqr" (case-insensitive).
    load_kwargs : forwarded to the solver's ``load`` (e.g. ``task=``, ``checkpoint=``,
        ``device=``, ``dtype=``).  Pass ``task="maze"`` to load the maze checkpoint
        (default ``task="sudoku"``); the solve / solve_batch API is identical either
        way -- only the puzzle codec and latent ``seq_len`` change.

    Examples
    --------
    >>> solver = get_solver("fprm")                 # Sudoku
    >>> solver = get_solver("eqr", task="maze")     # Maze
    >>> result = solver.solve(puzzle)
    """
    key = name.lower()
    if key not in _SOLVERS:
        raise ValueError(f"Unknown solver {name!r}. Available: {available_solvers()}")
    return _SOLVERS[key].load(**load_kwargs)
