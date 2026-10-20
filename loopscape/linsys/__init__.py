"""Looped transformer solving 8x8 linear systems over GF(3); weights from hf.co/GilpinLab/linsys-gf3."""
from . import basin, task
from .model import Config, LoopedTransformer
from .solver import LinsysSolver
