"""Linear-system solver: the GF(3) looped transformer behind the common ``Solver`` API.

``puzzle`` is a system ``(A, b)``; ``grid`` is the decoded answer ``x`` (a list of 8 ints) and
``solved`` checks ``A x = b`` over GF(3).  The model runs a fixed number of loops, so
``halted`` is always False.  Batched sweeps inject starting latents of shape ``(B, 97, 256)``.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from ..base import BatchSolveResult, Solver, SolveResult, register_solver
from . import basin
from .task import N, P, encode


class LinsysSolver(Solver):
    name = "linsys"

    @classmethod
    def load(cls, checkpoint: Optional[str] = None, device: str = "cpu", **_) -> "LinsysSolver":
        return cls(basin.load(checkpoint, device), device)

    @torch.no_grad()
    def _run(self, puzzle, z, B, loops, return_intermediates):
        A, b = puzzle
        idx = torch.tensor(encode(A, b), device=self.device).unsqueeze(0).expand(B, -1)
        hist = []
        on_iter = (lambda k, x: hist.append(self.model.decode(x[:, -N:]).argmax(-1) % P)) \
            if return_intermediates else None
        logits = self.model(idx, loops=loops, init_x=z, on_iter=on_iter)
        answers = (logits[:, -N:].argmax(-1) % P).cpu().numpy()
        solved = [bool(((np.asarray(A) @ x) % P == np.asarray(b) % P).all()) for x in answers]
        steps = [h.cpu().numpy().tolist() for h in hist] if return_intermediates else None
        return answers.tolist(), solved, steps

    def solve(self, puzzle, *, loops: int = 320, seed: Optional[int] = None,
              initial_latent: Optional[torch.Tensor] = None, return_intermediates: bool = False,
              **_) -> SolveResult:
        """One solve from the model's own random init (``seed``) or an injected ``(97, 256)`` latent."""
        if seed is not None:
            torch.manual_seed(seed)
        z = None if initial_latent is None else torch.as_tensor(initial_latent, device=self.device)[None]
        answers, solved, steps = self._run(puzzle, z, 1, loops, return_intermediates)
        return SolveResult(grid=answers[0], steps=loops, solved=solved[0], halted=False, solver=self.name,
                           intermediates=[s[0] for s in steps] if steps else None)

    def solve_batch(self, puzzle, *, initial_latents: torch.Tensor, loops: int = 320,
                    return_intermediates: bool = False, **_) -> BatchSolveResult:
        """Solve from B injected latents; ``intermediates`` is ``[steps][B]`` decoded answers."""
        z = torch.as_tensor(initial_latents, device=self.device)
        z = z[None] if z.ndim == 2 else z
        answers, solved, steps = self._run(puzzle, z, z.shape[0], loops, return_intermediates)
        return BatchSolveResult(grids=answers, steps=loops, solved=solved, halted=False,
                                solver=self.name, intermediates=steps)


register_solver("linsys", LinsysSolver)
