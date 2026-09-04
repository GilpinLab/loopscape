"""Countdown-SFT Parcae — the project's canonical Parcae checkpoint.

`GilpinLab/parcae-countdown-v1 <https://huggingface.co/GilpinLab/parcae-countdown-v1>`_
is a 140M Parcae fine-tuned (by us, on the upstream sandyresearch/parcae code)
on the Countdown task with a REDUCED integration step: ``dt_scale = 0.3``
multiplies the injection's learned per-channel ``dt``, slowing the recurrence so
trajectories settle over many more loops — the regime where basin structure is
visible.  **All Parcae analyses in this project use this checkpoint, never the
original SandyResearch releases.**  Prompts look like ``"1 14 20 57 77 -> 28\\n"``
(:func:`build_prompt`; the trailing newline matters) and are tokenized WITH BOS.

The repo mirrors the training run directory (``run_config.json``,
``tokenizer/``, ``checkpoints-*/step-*``) and is downloaded on first use into
the shared cache.  Set ``LOOPSCAPE_PARCAE_CD_SRC`` to point at a local run
directory instead.

Upstream ``parcae_lm`` (pinned; see :mod:`loopscape.parcae.download`) has no
``dt_scale`` — the field post-dates the pinned commit — so this module patches
``DiagonalInjection.forward`` at load time:

    dt = dt_scale * softplus(dt_bias)        # dt_scale = 1.0 -> upstream behavior

The basin recipe (mirrors the private campaign): baseline state
``h_bar = sigma * randn(T, d_rec)`` from the model's own init distribution, a
2-plane orthonormal to ``h_bar`` over ALL T positions
(:func:`context_plane_basis`), initial conditions ``h0 = h_bar + a*U + b*V``,
run ``M`` recurrence loops reading the next-token answer after every loop
(:func:`CountdownBasin.trace_answers`), and score each pixel by
loops-until-the-answer-stops-changing (:func:`settle_time_batch`; a pixel still
changing at loop ``M`` is censored).  Extents are quoted against
:func:`CountdownBasin.natural_norm` = ``sigma * sqrt(T * d_rec)``.
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import fields as dataclass_fields
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch
from torch import Tensor

from ..paths import CHECKPOINT_DIR
from .probe import _run_prelude  # importing .probe puts the pinned parcae source on sys.path

_CD_HF_REPO = "GilpinLab/parcae-countdown-v1"
_CD_CACHE = CHECKPOINT_DIR / "parcae_countdown_v1"


def build_prompt(nums: Sequence[int], target: int) -> str:
    """``([1, 14, 20], 28)`` -> ``"1 14 20 -> 28\\n"`` (trailing newline required)."""
    return " ".join(str(int(n)) for n in nums) + f" -> {int(target)}\n"


# ---------------------------------------------------------------------------
# Checkpoint acquisition
# ---------------------------------------------------------------------------

def ensure_countdown_run(timeout: float = 300.0) -> Path:
    """Local path to the Countdown run directory, downloading it if needed.

    ``LOOPSCAPE_PARCAE_CD_SRC`` overrides with a local training run directory
    (one containing ``run_config.json``, ``tokenizer/`` and a
    ``checkpoints-*/step-*`` weights file).
    """
    env = os.environ.get("LOOPSCAPE_PARCAE_CD_SRC")
    if env:
        p = Path(env).expanduser()
        if not (p / "run_config.json").is_file():
            raise FileNotFoundError(
                f"LOOPSCAPE_PARCAE_CD_SRC={env!r} has no run_config.json")
        return p
    if (_CD_CACHE / "run_config.json").is_file():
        return _CD_CACHE

    # huggingface_hub handles auth (private/gated repos), resume, and dedup.
    from huggingface_hub import snapshot_download

    try:
        snapshot_download(_CD_HF_REPO, local_dir=str(_CD_CACHE),
                          etag_timeout=timeout)
    except Exception as e:
        raise FileNotFoundError(
            f"Could not fetch hf.co/{_CD_HF_REPO} ({type(e).__name__}). If the "
            "repo is private, authenticate with `hf auth login` using a token "
            "that can read it; or set LOOPSCAPE_PARCAE_CD_SRC to a local run "
            "directory (run_config.json, tokenizer/, checkpoints-*/step-*)."
        ) from e
    if not (_CD_CACHE / "run_config.json").is_file():
        found = sorted(p.relative_to(_CD_CACHE).as_posix()
                       for p in _CD_CACHE.rglob("*") if p.is_file())[:10]
        raise FileNotFoundError(
            f"hf.co/{_CD_HF_REPO} does not contain run_config.json at its root; "
            f"found e.g. {found}. Expected a mirror of the training run dir."
        )
    return _CD_CACHE


def _find_checkpoint(run_dir: Path, step: Optional[int]) -> Path:
    """Newest (or requested) ``checkpoints-*/step-*`` weights file."""
    candidates = [p for p in run_dir.glob("checkpoints-*/step-*") if p.is_file()]
    if not candidates:
        raise FileNotFoundError(f"no checkpoints under {run_dir}/checkpoints-*/step-*")

    def step_of(p: Path) -> int:
        try:
            return int(p.name.split("step-")[1].split("-")[0])
        except (IndexError, ValueError):
            return 0

    if step is not None:
        for p in candidates:
            if step_of(p) == step:
                return p
        raise FileNotFoundError(
            f"step {step} not found; available: {sorted(step_of(p) for p in candidates)}")
    return max(candidates, key=step_of)


# ---------------------------------------------------------------------------
# dt_scale support on top of the pinned upstream parcae_lm
# ---------------------------------------------------------------------------

def _patch_dt_scale() -> None:
    """Make ``DiagonalInjection`` honor a ``dt_scale`` config attribute.

    Idempotent.  With no ``dt_scale`` attribute present (every stock Parcae
    config) the patched forward is bitwise-identical to upstream, so loading
    the released SandyResearch checkpoints through :mod:`.probe` is unaffected.
    """
    from parcae_lm.modules.injection import DiagonalInjection

    if getattr(DiagonalInjection, "_loopscape_dt_scale", False):
        return

    def forward(self, x_t: Tensor, e: Tensor) -> Tensor:
        dt = getattr(self.config, "dt_scale", 1.0) * torch.nn.functional.softplus(self.dt_bias)
        A = torch.exp(self.A_log)
        decay = torch.exp(-dt * A)
        return x_t * decay + dt * (e @ self.B.T)

    DiagonalInjection.forward = forward
    DiagonalInjection._loopscape_dt_scale = True


def load_countdown(
    run_dir: Optional[str] = None,
    *,
    step: Optional[int] = None,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
):
    """Build the Countdown-SFT Parcae from its run directory and load the weights.

    Returns ``(model, tokenizer, run_config)``.  The architecture is built from
    the run's own ``run_config.json`` (``model_name`` + ``model_overwrite``);
    fields the pinned upstream config predates (``dt_scale``) are applied to the
    injection modules directly, via :func:`_patch_dt_scale`.
    """
    from parcae_lm.models.config import Config as DynamicConfig
    from parcae_lm.models.parcae.config import ParcaeConfig
    from parcae_lm.tokenizer import Tokenizer
    from receval.models.parcae import ModelingParcae

    rd = Path(run_dir) if run_dir is not None else ensure_countdown_run()
    run_config = json.loads((rd / "run_config.json").read_text())
    overwrite = dict(run_config.get("model_overwrite", {}))

    known_fields = {f.name for f in dataclass_fields(ParcaeConfig)}
    extras = {k: overwrite.pop(k) for k in list(overwrite) if k not in known_fields}
    for k in extras:
        if k != "dt_scale":
            print(f"[countdown] WARNING: dropping unknown config field {k}={extras[k]!r} "
                  "(post-dates the pinned parcae_lm)")

    _patch_dt_scale()
    cfg = DynamicConfig.from_name(run_config["model_name"],
                                  **{**overwrite, "skip_initialization": True})
    model = ModelingParcae(cfg)
    if "dt_scale" in extras:
        for m in model.modules():
            if type(m).__name__ == "DiagonalInjection":
                m.config.dt_scale = float(extras["dt_scale"])

    ckpt = _find_checkpoint(rd, step)
    state = torch.load(ckpt, map_location="cpu", weights_only=False, mmap=True)
    payload = state["model"] if isinstance(state, dict) and "model" in state else state
    incompatible = model.load_state_dict(payload, strict=False)
    if incompatible.missing_keys:
        raise RuntimeError(
            f"checkpoint missing {len(incompatible.missing_keys)} keys, "
            f"e.g. {incompatible.missing_keys[:5]}")
    model = model.to(device=device, dtype=dtype).eval()

    tok_dir = rd / "tokenizer"
    tok = Tokenizer(str(tok_dir if tok_dir.is_dir()
                        else run_config.get("tokenizer_path", tok_dir)))
    return model, tok, run_config


# ---------------------------------------------------------------------------
# Basin geometry + tracing (ported from the private countdown campaign)
# ---------------------------------------------------------------------------

def context_plane_basis(h_bar: Tensor, seed: int):
    """Two orthonormal whole-state directions ``(T, d_rec)``, orthogonal to ``h_bar``."""
    T, d = h_bar.shape
    g = torch.Generator(device="cpu").manual_seed(seed + 20221)
    flat = h_bar.reshape(-1).float().cpu()
    r1 = torch.randn(T * d, generator=g)
    r2 = torch.randn(T * d, generator=g)
    cols = [r1, r2] if float(flat.norm()) == 0.0 else [flat, r1, r2]
    Q, _ = torch.linalg.qr(torch.stack(cols, dim=1))
    U = Q[:, -2].reshape(T, d).to(device=h_bar.device, dtype=h_bar.dtype)
    V = Q[:, -1].reshape(T, d).to(device=h_bar.device, dtype=h_bar.dtype)
    return U, V


def build_h0(h_bar: Tensor, alpha, beta, U: Tensor, V: Tensor) -> Tensor:
    """Initial conditions ``(B, T, d_rec)`` at plane coordinates ``(alpha, beta)``."""
    a = torch.as_tensor(alpha, device=h_bar.device, dtype=h_bar.dtype)[:, None, None]
    b = torch.as_tensor(beta, device=h_bar.device, dtype=h_bar.dtype)[:, None, None]
    return h_bar.unsqueeze(0) + a * U.unsqueeze(0) + b * V.unsqueeze(0)


def settle_time_batch(tokens: np.ndarray) -> np.ndarray:
    """Loops until the answer token stops changing; ``tokens`` is ``(B, M)``.

    A pixel whose loop-``M`` answer still differs from loop ``M-1`` scores ``M``
    (censored — treat as a lower bound).
    """
    B, M = tokens.shape
    differs = tokens != tokens[:, -1:]
    idx = np.arange(M)[None, :]
    last_diff = np.where(differs, idx, -1).max(axis=1)
    return (last_diff + 1).astype(np.int64)


class CountdownBasin:
    """Recurrence harness for basin sweeps on the Countdown-SFT checkpoint."""

    def __init__(self, model, tok, *, add_bos: bool = True, bos_id: int = 0):
        self.model, self.tok, self.add_bos = model, tok, add_bos
        self.bos_id = bos_id
        cfg = model.config
        self.cfg = cfg
        self.d_rec = cfg.recurrent_embedding_dimension
        self.mean_recurrence = cfg.mean_recurrence
        # state_init == "like-init": trunc_normal(std=sigma) * emb_scale
        self.sigma = float(cfg.init.get_std("embedding")) * float(cfg.init.embedding_scale)
        self.device = next(model.parameters()).device
        self.dtype = next(model.parameters()).dtype

    @classmethod
    def load(cls, run_dir: Optional[str] = None, *, step: Optional[int] = None,
             device: str = "cpu", dtype: torch.dtype = torch.float32) -> "CountdownBasin":
        model, tok, run_config = load_countdown(run_dir, step=step,
                                                device=device, dtype=dtype)
        # The run's own record of the effective special ids is the ground truth
        # (the tokenizer files alone may not define BOS).
        rd = Path(run_dir) if run_dir is not None else ensure_countdown_run()
        ids_path = rd / "special_token_ids.json"
        recorded = json.loads(ids_path.read_text()) if ids_path.is_file() else {}
        return cls(model, tok,
                   add_bos=bool(recorded.get("add_bos", run_config.get("add_bos", True))),
                   bos_id=int(recorded.get("bos_id", 0)))

    def prelude(self, prompt: str) -> Tensor:
        """Tokenize (WITH BOS, as trained) and run the prelude; sets ``self.T``."""
        ids = self.tok.encode(prompt, device=self.device, bos=False).long()
        if ids.dim() == 1:
            ids = ids.unsqueeze(0)
        if self.add_bos:
            bos = torch.full((ids.shape[0], 1), self.bos_id,
                             dtype=ids.dtype, device=ids.device)
            ids = torch.cat([bos, ids], dim=1)
        self.input_ids = ids
        self.T = ids.shape[1]
        with torch.no_grad():
            self.e, self.freqs_cis = _run_prelude(self.model, ids)
        self.model._current_input_ids = ids
        return self.e

    def baseline(self, seed: int) -> Tensor:
        """On-distribution baseline state ``h_bar`` of shape ``(T, d_rec)``."""
        g = torch.Generator(device="cpu").manual_seed(seed)
        h = torch.randn(self.T, self.d_rec, generator=g, dtype=torch.float32) * self.sigma
        return h.to(device=self.device, dtype=self.dtype)

    def natural_norm(self) -> float:
        """``||initialize_state(...)|| = sigma * sqrt(T * d_rec)`` — the unit for extents."""
        return self.sigma * math.sqrt(self.T * self.d_rec)

    # -- recurrence + readout (reference full-sequence path) --------------------
    def _ve(self, module_dict, idx: int, ids: Tensor) -> Optional[Tensor]:
        key = str(idx)
        return module_dict[key](ids) if key in module_dict else None

    def loop_full(self, h: Tensor) -> Tensor:
        """One recurrence loop on the full sequence, batched over ``h`` (B, T, d_rec)."""
        step = torch.tensor(0, device=h.device)
        total = torch.tensor(self.mean_recurrence, device=h.device)
        B = h.shape[0]
        e = self.e.expand(B, -1, -1)
        self.model._current_input_ids = self.input_ids.expand(B, -1)
        return self.model.update_recurrent_state(h, e, self.freqs_cis, None, step, total)

    def readout_full(self, h: Tensor) -> Tensor:
        """Coda + LM head on the final position only; returns logits ``(B, V)``."""
        m, cfg = self.model, self.cfg
        B = h.shape[0]
        ids = self.input_ids.expand(B, -1)
        x = m.transformer.C(h)
        coda_off = cfg.n_layers_in_prelude + cfg.n_layers_in_recurrent_block
        for i, block in enumerate(m.transformer.coda):
            ve = self._ve(m.value_embeds, coda_off + i, ids)
            x = block(x, self.freqs_cis, None, ve=ve)
        x = m.transformer.ln_f(x)[:, -1:, :]
        return self._head(x).squeeze(1)

    def _head(self, x: Tensor) -> Tensor:
        """LM head + logit scale + softcap, matching Parcae.forward's no-label branch."""
        m, cfg = self.model, self.cfg
        scale = cfg.init.logit_scale
        if getattr(cfg, "use_fused_head", None) == "full-triton":
            w = m.lm_head.weight.T if cfg.tie_embeddings else m.lm_head.weight
            logits = torch.matmul(x, w).float() * scale
        else:
            logits = m.lm_head(x).float() * scale
        if cfg.logit_softcap is not None:
            sc = cfg.logit_softcap
            logits = sc * torch.tanh(logits / sc)
        return logits

    @torch.no_grad()
    def trace_answers(self, h0: Tensor, M: int, batch: int = 256) -> np.ndarray:
        """``M`` loops from each initial condition, reading the answer after every loop.

        Returns the per-loop answer tokens ``(B, M) int64`` — feed to
        :func:`settle_time_batch`.  States that leave float range yield constant
        argmaxes; guard with higher ``M``/dtype if a field looks suspiciously flat.
        """
        B = h0.shape[0]
        toks = np.empty((B, M), dtype=np.int64)
        for s in range(0, B, batch):
            e = min(s + batch, B)
            h = h0[s:e].clone()
            for t in range(M):
                h = self.loop_full(h)
                toks[s:e, t] = self.readout_full(h).argmax(dim=-1).cpu().numpy()
        return toks
