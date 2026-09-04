"""Fetch and cache the HRM-Text-1B checkpoint from the Hugging Face Hub.

The weights (a single ~2.4 GB safetensors file, bfloat16) live at
``sapientinc/HRM-Text-1B``.  Downloaded with ``huggingface_hub.snapshot_download``
(resumable) and cached under ``downloaded_checkpoints/hrm-text-1b/`` so
subsequent loads are offline.  No modeling code is fetched: ``hrm_text`` is a
native transformers architecture (>= 5.9).
"""
from __future__ import annotations

import os

_REPO_ID = "sapientinc/HRM-Text-1B"
from ..paths import CHECKPOINT_DIR

_CACHE_DIR = str(CHECKPOINT_DIR / "hrm-text-1b")

_PATTERNS = ["config.json", "model.safetensors", "tokenizer*", "generation_config.json"]


def ensure_hrm_text_checkpoint(path: str | None = None, *, force: bool = False) -> str:
    """Return a local directory with the HRM-Text checkpoint, downloading if needed."""
    dest = path or _CACHE_DIR
    if not force and os.path.exists(os.path.join(dest, "model.safetensors")):
        return dest

    from huggingface_hub import snapshot_download

    snapshot_download(_REPO_ID, local_dir=dest, allow_patterns=_PATTERNS)
    return dest
