<!-- # loopscape -->

![loopscape](./assets/name.png)

Tools for probing the landscape of fractal basins in recurrent reasoning models

## Quickstart

Every solver has the same API: pick a model, pass it a puzzle string (such as an 81-char Sudoku puzzle, or a 900-char maze), and optionally inject your own initial latent
states:

```python
import torch
from loopscape import get_solver

puzzle = ".....6.....7.3.4..5..8.....9.8.7.3...1.....9...498.7....2....4387....2......2...."

solver = get_solver("eqr")                         # or "fprm"
z = torch.randn(64, 97, 512)                       # 64 initial conditions
r = solver.solve_batch(puzzle, z_H=z, z_L=z * 0,
                       max_steps=24, noise_scale=0.0,   # deterministic map
                       return_intermediates=True)
# r.grids, r.solved, r.intermediates ([steps][64] decoded grids)
```

FPRM takes `initial_latents=` instead of `z_H`/`z_L`, with residual halting
(`fp_thresh`, `max_iters`). The latent shapes vary by model and task: Sudoku has shape `(97, 512)` for both models, while
maze has shape `(916, 512)` for FPRM and shape `(916, 128)` for EqR.

**`demos/basins.ipynb`** producesbasins through a random 2D slice of each model's latent space and plots the resulting basin and settling-time fields for several (model, task) pairs.

## Install

```bash
uv add "loopscape @ git+https://github.com/williamgilpin/loopscape"
# + matplotlib/jupyter, to run demos/basins.ipynb:
uv add "loopscape[demos] @ git+https://github.com/williamgilpin/loopscape"
# or from a local checkout:
uv add --editable path/to/loopscape
```

Requires Python ≥ 3.11. On first use each solver downloads its weights. For EqR/Parcae, the upstream model code is pinned to a fixed commit — into a shared cache (`downloaded_checkpoints/` at the repo root; override with `LOOPSCAPE_CACHE_DIR`). On offline clusters, warm the cache first:

```bash
uv run python -c "from loopscape import get_solver; get_solver('eqr')"
```

<!-- ## Layout

The **`loopscape/`** directory contains a unified wrapper to several recurrent reasoning models, allowing them to be used interchangeably for inference and analysis.

| module | contents |
|---|---|
| `loopscape.base` | `Solver` API, `SolveResult`/`BatchSolveResult`, the `get_solver` registry |
| `loopscape.tasks` | Sudoku / Maze codecs shared by all solvers |
| `loopscape.fprm` | FPRM solver (arXiv:2606.18206) |
| `loopscape.eqr` | EqR solver (arXiv:2605.21488) |
| `loopscape.parcae` | Parcae looped-LM latent probe |
| `loopscape.utils` | `batched_apply`, `load_pt_gz` |
| `loopscape.metrics` | basin entropy, uncertainty exponent, ordinal complexity, FLI |
| `loopscape.fractal` | register-subspace slice construction | -->

## Weights and upstream code

* **FPRM** — inference architecture reimplemented in-package; released
  checkpoints fetched from
  [fixed-point-reasoners/fprm](https://huggingface.co/fixed-point-reasoners/fprm).
* **EqR** — weights from
  [locuslab/EqR-model](https://huggingface.co/locuslab/EqR-model); model code
  from [locuslab/EqR](https://github.com/locuslab/EqR) pinned to commit
  `e9934826` (override with `LOOPSCAPE_EQR_SRC`).
* **Parcae** — model code from
  [sandyresearch/parcae](https://github.com/sandyresearch/parcae) pinned to
  commit `6e451955` (`LOOPSCAPE_PARCAE_SRC` overrides). Weights are **our own
  checkpoint**,
  [GilpinLab/parcae-countdown-v1](https://huggingface.co/GilpinLab/parcae-countdown-v1)
  — a 140M Parcae fine-tuned on Countdown with a reduced integration step
  (`dt_scale = 0.3`) — used for every Parcae analysis here (never the original
  SandyResearch releases); see `loopscape.parcae.countdown`.

![Basins in the convergence time on a Sudoku puzzle](./assets/sudoku_medium.png)