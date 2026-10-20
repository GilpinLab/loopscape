"""Fetch and cache FPRM weight checkpoints from the Hugging Face Hub.

The released FPRM training series lives at ``fixed-point-reasoners/fprm``
(``{task}/step_6510 .. step_78120``, 12 checkpoints per task, plus an
``all_config.yaml`` sidecar).  Checkpoints are cached under
``paths.FPRM_WEIGHTS_DIR / {task}/`` so subsequent loads are offline.  Only
``requests`` is needed -- no ``huggingface_hub`` dependency.
"""
from __future__ import annotations

import os

import requests

from ..paths import FPRM_WEIGHTS_DIR

HF_REPO = "fixed-point-reasoners/fprm"
FINAL_STEP = 78120


def ensure_fprm_checkpoint(task: str = "sudoku", step: int = FINAL_STEP, *,
                           force: bool = False, timeout: float = 120.0) -> str:
    """Return a local path to the FPRM ``step_{step}`` checkpoint for ``task``.

    Downloads from the released series on the Hub on a cache miss.
    """
    dest = os.path.join(str(FPRM_WEIGHTS_DIR), task, f"step_{step}")
    if os.path.exists(dest) and not force:
        return dest

    os.makedirs(os.path.dirname(dest), exist_ok=True)
    url = f"https://huggingface.co/{HF_REPO}/resolve/main/{task}/step_{step}"
    tmp = dest + ".part"
    with requests.get(url, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):  # 1 MB chunks
                if chunk:
                    f.write(chunk)
    os.replace(tmp, dest)  # atomic: only appears complete once fully written
    return dest
