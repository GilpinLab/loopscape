"""Fetch and cache the Huginn-0125 checkpoint from the Hugging Face Hub.

The weights (~15.6 GB across 4 safetensors shards, stored in float32) live at
``tomg-group-umd/huginn-0125``.  Unlike EqR's single-file download this uses
``huggingface_hub.snapshot_download`` (already a transformers dependency): it is
resumable, verifies file sizes, and also fetches the tokenizer files.  Everything
is cached under ``downloaded_checkpoints/huginn-0125/`` so subsequent loads are
offline.
"""
from __future__ import annotations

import os

_REPO_ID = "tomg-group-umd/huginn-0125"
from ..paths import CHECKPOINT_DIR

_CACHE_DIR = str(CHECKPOINT_DIR / "huginn-0125")

# Weights + tokenizer + configs; the raven_*.py modeling files are also kept for
# provenance (the copies actually imported are vendored in fprm/solvers/_huginn).
_PATTERNS = [
    "*.safetensors",
    "model.safetensors.index.json",
    "config.json",
    "generation_config.json",
    "tokenizer*",
    "special_tokens_map.json",
    "raven_*.py",
]


def ensure_huginn_checkpoint(path: str | None = None, *, force: bool = False) -> str:
    """Return a local directory containing the Huginn checkpoint, downloading if needed.

    Parameters
    ----------
    path  : explicit local checkpoint directory.  If given and it already holds a
            ``model.safetensors.index.json``, it is used as-is; if given and
            missing/incomplete, the snapshot is downloaded there.
    force : re-verify/re-download the snapshot even if the index file exists.
    """
    dest = path or _CACHE_DIR
    if not force and os.path.exists(os.path.join(dest, "model.safetensors.index.json")):
        return dest

    from huggingface_hub import snapshot_download

    snapshot_download(_REPO_ID, local_dir=dest, allow_patterns=_PATTERNS)
    return dest
