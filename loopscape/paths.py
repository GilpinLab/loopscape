"""Where loopscape resolves its on-disk cache, in one place.

Every downloader (model weights, pinned upstream source snapshots) caches under
``CHECKPOINT_DIR``.  Resolution order:

1. ``LOOPSCAPE_CACHE_DIR`` environment variable, if set.
2. An existing ``downloaded_checkpoints/`` next to this repo's root
   (``<loopscape repo>/downloaded_checkpoints``).
3. An existing ``downloaded_checkpoints/`` one level higher -- the layout when
   loopscape is checked out inside a host repo that already owns a cache.
4. Otherwise, (1)'s default: ``<loopscape repo>/downloaded_checkpoints`` is
   created on first download.

This module imports only ``os``/``pathlib``: it is a leaf module, safe to import
from anywhere in the package.
"""
import os
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent


def _resolve_cache_dir() -> Path:
    env = os.environ.get("LOOPSCAPE_CACHE_DIR")
    if env:
        return Path(env).expanduser()
    candidates = [
        PACKAGE_DIR.parent / "downloaded_checkpoints",         # <repo>/downloaded_checkpoints
        PACKAGE_DIR.parent.parent / "downloaded_checkpoints",  # nested inside a host repo
    ]
    for c in candidates:
        if c.is_dir():
            return c
    return candidates[0]


CHECKPOINT_DIR = _resolve_cache_dir()

# FPRM weight checkpoints (the released series from hf.co/fixed-point-reasoners/fprm)
# live under the shared cache; see loopscape.fprm.download.
FPRM_WEIGHTS_DIR = CHECKPOINT_DIR / "fprm_weights"

# Pinned upstream source snapshots (EqR, Parcae) are extracted here; see
# loopscape.external.
SOURCE_DIR = CHECKPOINT_DIR / "_src"
