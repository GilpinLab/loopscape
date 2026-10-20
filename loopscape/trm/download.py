"""Fetch and cache the TRM Maze-Hard checkpoint from the Hugging Face Hub.

Samsung SAIL Montreal released the TRM code but no weights; the best public
checkpoint is alphaXiv's independent reproduction (``alphaXiv/trm-model-maze``,
file ``maze_hard_step_32550``, ~26 MB): a full 32550-step run of the upstream
README's Maze-Hard recipe scoring 83.67% +- 2.28% exact accuracy on the test
split (paper: 85.3%).  Downloaded EqR-style with ``requests`` and cached under
``downloaded_checkpoints/``.
"""
from __future__ import annotations

import os

import requests

_HF_URL = {
    "maze": "https://huggingface.co/alphaXiv/trm-model-maze/resolve/main/maze_hard_step_32550",
}
_CACHE_NAME = {"maze": "trm_maze.pt"}

from ..paths import CHECKPOINT_DIR

_CACHE_DIR = str(CHECKPOINT_DIR)


def ensure_trm_checkpoint(path: str | None = None, *, task: str = "maze",
                          force: bool = False, timeout: float = 120.0) -> str:
    """Return a local path to the TRM checkpoint for ``task``, downloading if needed.

    Mirrors :func:`loopscape.eqr.download.ensure_eqr_checkpoint`.
    """
    if task not in _HF_URL:
        raise ValueError(f"No public TRM checkpoint for task {task!r}; "
                         f"available: {sorted(_HF_URL)}")
    dest = path or os.path.join(_CACHE_DIR, _CACHE_NAME[task])
    if os.path.exists(dest) and not force:
        return dest

    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    tmp = dest + ".part"
    with requests.get(_HF_URL[task], stream=True, timeout=timeout) as r:
        r.raise_for_status()
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                if chunk:
                    f.write(chunk)
    os.replace(tmp, dest)
    return dest
