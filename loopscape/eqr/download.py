"""Fetch and cache the EqR Sudoku checkpoint from the Hugging Face Hub.

The EqR weights are not bundled in the repo; they live at
``locuslab/EqR-model`` (file ``sudoku-extreme/eqr.pth``, ~40 MB).  We download
the file via a direct ``resolve/`` URL with ``requests`` (already a dependency)
and cache it under the repo's ``downloaded_checkpoints/`` so subsequent loads are
offline.  No ``huggingface_hub`` dependency is required.
"""
from __future__ import annotations

import os

import requests

# Per-task EqR checkpoint on the Hub (locuslab/EqR-model) and its local cache name.
_HF_URL = {
    "sudoku": "https://huggingface.co/locuslab/EqR-model/resolve/main/sudoku-extreme/eqr.pth",
    "maze": "https://huggingface.co/locuslab/EqR-model/resolve/main/maze-unique/eqr.pth",
}
_CACHE_NAME = {"sudoku": "eqr_sudoku.pth", "maze": "eqr_maze.pth"}

from ..paths import CHECKPOINT_DIR

_CACHE_DIR = str(CHECKPOINT_DIR)


def ensure_eqr_checkpoint(path: str | None = None, *, task: str = "sudoku",
                          force: bool = False, timeout: float = 120.0) -> str:
    """Return a local path to the EqR checkpoint for ``task``, downloading if needed.

    Parameters
    ----------
    path    : explicit local path.  If given and it exists, it is used as-is; if
              given and missing, the download is written there.
    task    : "sudoku" | "maze" -- selects which released EqR checkpoint to fetch.
    force   : re-download even if the cached file exists.
    timeout : per-request connect/read timeout in seconds.
    """
    if task not in _HF_URL:
        raise ValueError(f"Unknown EqR task {task!r}; expected one of {sorted(_HF_URL)}")
    dest = path or os.path.join(_CACHE_DIR, _CACHE_NAME[task])
    if os.path.exists(dest) and not force:
        return dest

    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    tmp = dest + ".part"
    with requests.get(_HF_URL[task], stream=True, timeout=timeout) as r:
        r.raise_for_status()
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):  # 1 MB chunks
                if chunk:
                    f.write(chunk)
    os.replace(tmp, dest)  # atomic: only appears complete once fully written
    return dest


# Pinned upstream EqR source (github.com/locuslab/EqR). The solver imports
# ``models.eqr`` from this snapshot, so results cannot drift with upstream HEAD.
# Bump deliberately, and re-verify solver outputs when you do.
_EQR_REPO = "locuslab/EqR"
_EQR_COMMIT = "e993482666047339085333dfa3d502c1d958d204"  # 2026-08-04


def ensure_eqr_source() -> str:
    """Local path to the pinned EqR source tree, downloading it if needed.

    Override with the ``LOOPSCAPE_EQR_SRC`` environment variable to use a
    local checkout (e.g. a modified copy used for training runs).
    """
    from ..external import ensure_github_source
    return ensure_github_source(_EQR_REPO, _EQR_COMMIT, name="EqR")
