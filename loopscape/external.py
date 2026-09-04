"""Fetch pinned snapshots of upstream GitHub repos this package depends on.

loopscape deliberately does NOT vendor upstream model code (EqR, Parcae).
Instead each dependency is pinned to a specific commit and downloaded once as a
GitHub tarball into ``paths.SOURCE_DIR``, so upstream changes can never silently
alter results.  The extracted directory is what solvers put on ``sys.path``.

Offline environments (e.g. cluster compute nodes): run the relevant
``get_solver(...)`` once on a login node to warm the cache; subsequent loads are
purely local.  A ``LOOPSCAPE_<NAME>_SRC`` environment variable overrides the
download entirely and points at a local checkout (useful for developing against
a modified copy).
"""
from __future__ import annotations

import os
import shutil
import tarfile
import tempfile
from pathlib import Path

import requests

from .paths import SOURCE_DIR


def ensure_github_source(repo: str, commit: str, *, name: str | None = None,
                         timeout: float = 120.0) -> str:
    """Return a local path to ``repo`` at ``commit``, downloading if needed.

    Parameters
    ----------
    repo   : "owner/name" GitHub repository.
    commit : full or abbreviated commit SHA (or a tag). Pin a full SHA in
             calling code so results are reproducible.
    name   : cache directory label (default: the repo name).

    The environment variable ``LOOPSCAPE_{NAME}_SRC`` (upper-cased ``name``)
    overrides the pinned download with a local checkout.
    """
    label = (name or repo.split("/")[-1])
    env = os.environ.get(f"LOOPSCAPE_{label.upper().replace('-', '_')}_SRC")
    if env:
        if not os.path.isdir(env):
            raise FileNotFoundError(
                f"LOOPSCAPE_{label.upper()}_SRC={env!r} is not a directory")
        return env

    dest = SOURCE_DIR / f"{label}-{commit[:12]}"
    if dest.is_dir():
        return str(dest)

    url = f"https://codeload.github.com/{repo}/tar.gz/{commit}"
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=SOURCE_DIR) as tmp:
        tarball = Path(tmp) / "src.tar.gz"
        with requests.get(url, stream=True, timeout=timeout) as r:
            r.raise_for_status()
            with open(tarball, "wb") as f:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    if chunk:
                        f.write(chunk)
        with tarfile.open(tarball, "r:gz") as tf:
            tf.extractall(tmp, filter="data")
        # The tarball contains a single "<name>-<ref>/" root directory.
        roots = [p for p in Path(tmp).iterdir() if p.is_dir()]
        if len(roots) != 1:
            raise RuntimeError(f"Unexpected tarball layout for {repo}@{commit}: {roots}")
        shutil.move(str(roots[0]), str(dest))  # atomic-enough: appears once complete
    return str(dest)
