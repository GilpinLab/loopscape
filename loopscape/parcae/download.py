"""Pinned upstream Parcae source (github.com/sandyresearch/parcae).

The probe imports ``parcae_lm`` from this snapshot; pinning the commit keeps
results reproducible if upstream moves.  Override with the
``LOOPSCAPE_PARCAE_SRC`` environment variable to use a local checkout.
"""
from __future__ import annotations

_PARCAE_REPO = "sandyresearch/parcae"
_PARCAE_COMMIT = "6e4519556e5a9d444793c56aa9fd9ac2965f097b"


def ensure_parcae_source() -> str:
    from ..external import ensure_github_source
    return ensure_github_source(_PARCAE_REPO, _PARCAE_COMMIT, name="parcae")
