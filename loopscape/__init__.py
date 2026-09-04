"""loopscape -- probing basins of attraction in looped reasoning models.

A unified, pick-your-model API over several recurrent/looped reasoning models,
plus the latent-slice utilities and basin/complexity metrics used to analyse
them:

    from loopscape import get_solver
    solver = get_solver("fprm")          # or "eqr", "trm", "huginn", "hrm-text"
    result = solver.solve(puzzle)
    print(result.grid, result.solved, result.steps)

Submodules
----------
* ``base``     : the ``Solver`` ABC, result dataclasses, and the solver registry.
* ``tasks``    : puzzle codecs (Sudoku, Maze) shared by all solvers.
* ``fprm``     : FPRM architecture + solver (weights fetched from HF on demand).
* ``eqr``      : EqR solver; upstream model code is a pinned GitHub dependency.
* ``trm``, ``huginn``, ``hrm_text`` : further solvers (vendored minimal arch or
  native transformers; weights fetched on demand).
* ``parcae``   : latent-probing helpers for the Parcae LM (pinned dependency).
* ``utils``    : ``batched_apply`` and friends -- slice/trace plumbing.
* ``metrics``  : basin entropy, uncertainty exponent, ordinal complexity, FLI.
* ``fractal``  : register-subspace slice construction and raw recurrences.

Problem/dataset loaders are deliberately NOT part of this package -- supply
puzzle strings from your own data pipeline.
"""
from .base import (
    Solver,
    SolveResult,
    BatchSolveResult,
    get_solver,
    available_solvers,
    register_solver,
)
from .tasks import Task, SUDOKU, MAZE, get_task, available_tasks, maze_shortest_path_length
from .fprm.config import FPTRMConfig
from .fprm.inference import (
    load_pretrained,
    solve_sudoku,
    solve_sudoku_batch,
    BatchSolveInfo,
    SolveInfo,
    encode_puzzle,
    decode_solution,
    parse_puzzle,
    format_grid,
    is_valid_solution,
)
# Importing the solver modules registers them with the factory.
from .fprm.solver import FPRMSolver
from .eqr.solver import EqRSolver
from .huginn.solver import HuginnSolver
from .hrm_text.solver import HRMTextSolver
from .trm.solver import TRMSolver
from . import metrics, utils

__all__ = [
    "Solver",
    "SolveResult",
    "BatchSolveResult",
    "get_solver",
    "available_solvers",
    "register_solver",
    "Task",
    "SUDOKU",
    "MAZE",
    "get_task",
    "available_tasks",
    "maze_shortest_path_length",
    "FPTRMConfig",
    "load_pretrained",
    "solve_sudoku",
    "solve_sudoku_batch",
    "BatchSolveInfo",
    "SolveInfo",
    "encode_puzzle",
    "decode_solution",
    "parse_puzzle",
    "format_grid",
    "is_valid_solution",
    "FPRMSolver",
    "EqRSolver",
    "HuginnSolver",
    "HRMTextSolver",
    "TRMSolver",
    "metrics",
    "utils",
]
