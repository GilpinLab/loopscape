"""EqR (Equilibrium Reasoners, arXiv:2605.21488) solver.

Upstream model code is a PINNED GitHub dependency (locuslab/EqR), downloaded on
first use -- see :mod:`loopscape.external`. Weights come from the Hub
(locuslab/EqR-model).
"""
from .solver import EqRSolver
