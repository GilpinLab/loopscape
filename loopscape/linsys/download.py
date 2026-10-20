"""Fetch and cache the GF(3) linear-system checkpoint (hf.co/GilpinLab/linsys-gf3).

``LOOPSCAPE_LINSYS_CKPT`` points at a local state dict instead, e.g. one from ``train.py``.
"""
from __future__ import annotations

import os

import requests

from ..paths import CHECKPOINT_DIR

_HF_REPO = "GilpinLab/linsys-gf3"
_HF_FILE = "model.pt"
_HF_URL = f"https://huggingface.co/{_HF_REPO}/resolve/main/{_HF_FILE}"
_CACHE = os.path.join(str(CHECKPOINT_DIR), "linsys_gf3.pt")


def ensure_linsys_checkpoint(path: str | None = None, *, force: bool = False,
                             timeout: float = 120.0) -> str:
    """Return a local path to the checkpoint, downloading it on first use."""
    env = os.environ.get("LOOPSCAPE_LINSYS_CKPT")
    if path is None and env:
        return env
    dest = path or _CACHE
    if os.path.exists(dest) and not force:
        return dest
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    tmp = dest + ".part"
    with requests.get(_HF_URL, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                if chunk:
                    f.write(chunk)
    os.replace(tmp, dest)
    return dest
