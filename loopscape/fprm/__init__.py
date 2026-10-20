"""FPRM (Fixed-Point Reasoning Model) -- architecture, inference, and solver.

Reconstructs the inference path of "Fixed-Point Reasoners: Stable and Adaptive
Deep Looped Transformers" (arXiv:2606.18206). The architecture config ships in
this package (``all_config.yaml``); weight checkpoints are fetched on demand
from hf.co/fixed-point-reasoners/fprm into the shared cache (see
:mod:`loopscape.paths`).
"""
from .config import FPTRMConfig
from .inference import (
    load_pretrained, solve_sudoku, solve_sudoku_batch, BatchSolveInfo, SolveInfo,
    encode_puzzle, decode_solution, parse_puzzle, format_grid, is_valid_solution,
)
from .solver import FPRMSolver
