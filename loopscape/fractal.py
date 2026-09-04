"""Fractal settling-time structure in the FPRM maze model: field, statistics, zoom chase.

Self-contained.  Replaces a chain of nineteen modules that had accreted around this
measurement, of which two carried 1195 lines to supply 139 that were actually used
(`maze_prefix_basins.py` 752 lines for one class and two functions; `overnight_search.py`
443 lines for what reduces to a single `torch.zeros`).

Everything needed to reproduce the result is here:

    Recurrence            the FPRM maze map for one fixed maze, fp64
    plane_basis           orthonormal 2-plane in the register subspace, RMS-calibrated
    settle_field          the observable, optionally sharded across GPUs
    variogram_H           the roughness statistic, censor-masked and saturation-guarded
    box_D                 box dimension WITH the smooth-curve null band at that resolution
    chase                 nested zoom, re-centring on the roughest window at each level

Run `python fractal.py --help` for the CLI, or import the pieces.

THE OBSERVABLE
--------------
    r_i = || z_i - z_{i+1} ||_inf / || z_{i+1} ||_inf     sup over all tokens and channels
    T   = min { i : r_i < tau }                           tau = 0.05, integer, fp64

This is the standard stopping rule for a fixed-point iteration (the paper's `fp_thresh`),
NOT control-theory settling time: a step size rather than a distance to z*, and a first
crossing rather than enter-and-remain.  Those differ in principle; they were measured to
agree here (12% of pixels re-exit above tau after first crossing, yet H moves by <0.005).

`tau` selects which part of the transient is measured and is the single most important
knob.  At 1e-3 the trajectory is deep in the linear regime around the fixed point, where
the approach is governed by one eigenvalue and is smooth by construction.  At 0.05 it is
still in the nonlinear transient, which is where the folding lives.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time

import numpy as np
import torch

from .base import get_solver
from .tasks import MAZE

PREFIX_LEN = 16          # puzzle_emb_len: the register block
N_TOK = 916              # 16 prefix + 900 grid cells
D = 512                  # hidden size
REGISTERS = (1, PREFIX_LEN)   # tokens 1..15 receive exactly zero input injection


# ------------------------------------------------------------------ the map
class Recurrence:
    """The FPRM maze recurrence for one maze, at a chosen precision."""

    def __init__(self, question: str, device="cuda:0", dtype=torch.float64):
        self.device, self.dtype = device, dtype
        # (ported note: was `get_solver("fprm").load(...)`, which loaded and
        # discarded the Sudoku checkpoint before loading maze; same classmethod,
        # same result, one load instead of two.)
        self.model = get_solver("fprm", device=device, dtype=dtype, task="maze")
        self.inner = self.model.model.inner
        cos, sin = self.inner.rotary_emb()
        self.seq_info = dict(cos_sin=(cos[:N_TOK], sin[:N_TOK]),
                             puzzle_emb_len=PREFIX_LEN)
        self.set_question(question)

    def set_question(self, question: str):
        """Point the same loaded weights at a different maze."""
        self.question = question
        self._tokens = MAZE.encode(question).to(self.device)
        self._inj_cache: dict[int, torch.Tensor] = {}

    def injection(self, B: int) -> torch.Tensor:
        """The frozen input injection x, batched to B (cached per batch size)."""
        if B not in self._inj_cache:
            ident = torch.zeros((B,), dtype=torch.int32, device=self.device)
            self._inj_cache[B] = self.inner._input_embeddings(
                self._tokens.repeat(B, 1), ident)
        return self._inj_cache[B]


def plane_basis(seed: int = 0, subspace=REGISTERS):
    """Base point z0 = 0 and an orthonormal 2-plane inside the register subspace.

    Scaled so a plane coordinate of 1 displaces each perturbed token by RMS 1.  Since
    `norm_placement="output"` pins every latent token to RMS 1, `radius` is then directly
    the *fractional* size of the perturbation, and is comparable across subspaces of
    different dimension.
    """
    lo, hi = subspace
    n_tok = hi - lo
    g = torch.Generator().manual_seed(seed)
    Q, _ = torch.linalg.qr(torch.randn(n_tok * D, 2, generator=g, dtype=torch.float64))
    Q = Q * (n_tok * D) ** 0.5
    base = torch.zeros(N_TOK, D, dtype=torch.float64)
    u = torch.zeros(N_TOK, D, dtype=torch.float64)
    v = torch.zeros(N_TOK, D, dtype=torch.float64)
    u[lo:hi] = Q[:, 0].reshape(n_tok, D)
    v[lo:hi] = Q[:, 1].reshape(n_tok, D)
    return base, u, v


def _third_direction(u, v):
    """A deterministic third basis vector for the sphere chart: unit direction in
    the same token support as (u, v), orthogonal to both, RMS-calibrated like
    them (per-element RMS 1, i.e. global norm sqrt(n_tok * D))."""
    support = (u.abs().sum(-1) + v.abs().sum(-1)) > 0
    n_tok, Dh = int(support.sum()), u.shape[-1]
    g = torch.Generator().manual_seed(20260813)
    w = torch.randn(n_tok * Dh, generator=g, dtype=torch.float64)
    uu = u[support].reshape(-1); uu = uu / uu.norm()
    vv = v[support].reshape(-1); vv = vv / vv.norm()
    w = w - (w @ uu) * uu - (w @ vv) * vv
    w = w / w.norm() * (n_tok * Dh) ** 0.5
    t = torch.zeros_like(u)
    t[support] = w.reshape(n_tok, Dh)
    return t


def points_to_latents(base, u, v, coords, radius, normalize=False):
    """Plane coordinates -> latents.  Default: z = base + radius*(x*u + y*v).

    normalize=True maps the mesh onto the surface of a sphere instead (the
    exponential map): mesh radius |c| becomes the polar angle (radians) away
    from a third orthonormal register direction t, mesh angle the azimuth in
    the (u, v) plane,

        z = base + radius * ( cos|c| * t  +  sin|c|/|c| * (x*u + y*v) ).

    Every sample then has global perturbation norm EXACTLY radius*sqrt(n_tok*D)
    -- no amplitude gradient across the field, so settling time carries no
    initial-scale trend (Theorem 2: T grows with log of the injected
    perturbation size).  Distinct mesh points stay distinct for |c| < pi, and
    the map is smooth at the origin (the centre pixel is the t direction).
    Assumes the recipe's base = 0; a nonzero base is added afterwards and will
    reintroduce norm variation.
    """
    c = coords.to(torch.float64)
    if not normalize:
        return base + radius * (c[:, 0, None, None] * u + c[:, 1, None, None] * v)
    t = _third_direction(u, v)
    s = c.norm(dim=1)
    sinc = torch.where(s > 0, torch.sin(s) / s.clamp_min(1e-300), torch.ones_like(s))
    w = c[:, 0, None, None] * u + c[:, 1, None, None] * v
    return base + radius * (torch.cos(s)[:, None, None] * t + sinc[:, None, None] * w)


def grid(center=(0.0, 0.0), width=1.0, res=128):
    cx, cy = center
    lx = torch.linspace(cx - width, cx + width, res, dtype=torch.float64)
    ly = torch.linspace(cy - width, cy + width, res, dtype=torch.float64)
    gy, gx = torch.meshgrid(ly, lx, indexing="ij")
    return torch.stack([gx.ravel(), gy.ravel()], -1)


# ------------------------------------------------------------- the observable
@torch.no_grad()
def settle_field(rec, base, u, v, coords, radius=0.3, eta=1.0, n_max=1500, tau=0.05,
                 chunk=256, compact_every=20, window=0, label=""):
    """Settling time per plane coordinate.

    `window > 0` switches to persistence semantics: T is the start of the first run of
    `window` consecutive iterations under tau, rather than the first crossing.  Also
    returns the per-pixel count of re-exits above tau after the first crossing, which is
    the direct measure of how much the two definitions can differ.

    Settled trajectories are compacted out of the batch every `compact_every` steps, so
    raising `n_max` costs almost nothing on fields that mostly settle quickly.
    """
    N = coords.shape[0]
    T = np.full(N, n_max, dtype=np.int64)
    RX = np.zeros(N, dtype=np.int64)
    t0 = time.time()
    for i in range(0, N, chunk):
        cs = coords[i:i + chunk]
        B = cs.shape[0]
        z = points_to_latents(base, u, v, cs, radius).to(rec.device, rec.dtype)
        x = rec.injection(B)
        alive = torch.arange(B, device=rec.device)
        first = torch.full((B,), n_max, dtype=torch.long, device=rec.device)
        stay = torch.full((B,), n_max, dtype=torch.long, device=rec.device)
        run = torch.zeros(B, dtype=torch.long, device=rec.device)
        reexit = torch.zeros(B, dtype=torch.long, device=rec.device)
        was = torch.zeros(B, dtype=torch.bool, device=rec.device)
        for it in range(n_max):
            zb = rec.inner.L_level(z, x[:z.shape[0]], **rec.seq_info)
            r = ((z - zb).abs().amax(dim=(1, 2))
                 / (zb.abs().amax(dim=(1, 2)) + 1e-30)).double()
            under = r < tau
            idx = alive
            hit = under & (first[idx] == n_max)
            if bool(hit.any()):
                first[idx[hit]] = it
            reexit[idx] += (was[idx] & ~under & (first[idx] < n_max)).long()
            was[idx] = under
            run[idx] = torch.where(under, run[idx] + 1, torch.zeros_like(run[idx]))
            if window:
                done = (run[idx] >= window) & (stay[idx] == n_max)
                if bool(done.any()):
                    stay[idx[done]] = it - window + 1
            z = zb if eta == 1.0 else (1.0 - eta) * z + eta * zb
            z = torch.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
            if (it + 1) % compact_every == 0:
                target = stay if window else first
                keep = target[alive] == n_max
                if not bool(keep.any()):
                    break
                if int(keep.sum()) < z.shape[0]:
                    z = z[keep].contiguous()
                    alive = alive[keep]
        out = (stay if window else first).cpu().numpy()
        T[i:i + B] = out
        RX[i:i + B] = reexit.cpu().numpy()
        if (i // chunk) % 20 == 0 or i + B >= N:
            el = time.time() - t0
            print(f"    {label}{min(i + chunk, N)}/{N} px  {el / 60:.1f} min "
                  f"(eta {el / (i + B) * (N - i - B) / 60:.1f} min)", flush=True)
    return T, RX


def settle_field_sharded(recs, *a, label="", **kw):
    """Split one field's pixels across several GPUs, interleaved.

    Interleaved (`coords[i::n]`), never contiguous halves: settling time varies smoothly
    across the plane, so halves would hand one GPU the fast region and the other the slow
    one and the field would take as long as its slowest half.  Verified to reproduce the
    single-GPU result digit-for-digit.
    """
    n = len(recs)
    if n == 1:
        return settle_field(recs[0], *a, label=label, **kw)
    coords = a[3]
    outs, errs = [None] * n, [None] * n

    def run(i):
        try:
            outs[i] = settle_field(recs[i], a[0], a[1], a[2], coords[i::n], *a[4:],
                                   label=(f"{label}[gpu{i}] " if i == 0 else ""), **kw)
        except Exception as e:                                   # noqa: BLE001
            errs[i] = e

    ts = [threading.Thread(target=run, args=(i,), daemon=True) for i in range(n)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    for e in errs:
        if e is not None:
            raise e
    N = coords.shape[0]
    T = np.empty(N, dtype=np.int64)
    RX = np.empty(N, dtype=np.int64)
    for i in range(n):
        T[i::n], RX[i::n] = outs[i]
    return T, RX


# ------------------------------------------------------------------ statistics
def variogram_H(T, n_max, ks=(1, 2, 4, 8, 16, 32), sat_tol=0.5, r2_min=0.94):
    """Roughness exponent, censor-masked and saturation-guarded.

    H = 1 is a smooth differentiable field; H < 1 is self-affine roughness whose level
    sets have dimension 2 - H.  H needs no threshold, no box counting and no null band,
    which makes it far more reliable here than a box dimension.

    CENSORING.  A pixel that never crosses tau is stored as `n_max`, a clipped value.
    Clipped neighbour pairs contribute |dT| = 0 where the true difference is unknown and
    probably large, biasing H *upward* toward "smooth".  Only pairs with both pixels
    uncensored are fitted.

    SATURATION.  A fit hides the curve it came from.  Genuine roughness has mean|dT| at
    one pixel small next to the field's own median, growing steadily.  A saturated field
    has neighbours already as different as distant pixels: the variogram is flat from
    k = 1 and H is pinned near zero with no correlation left to decay -- decorrelation,
    not roughness, which a heavy tail in T produces.  One maze here returned H = 0.067
    with mean|dT| at k = 1 equal to 3.3x its median T.  Above `sat_tol` H is returned as
    NaN rather than as a number that will be read as a fractal.

    Returns (H, H_raw, frac_censored, saturation_ratio, R^2 of the log-log fit).
    """
    T = np.asarray(T, float)
    ok = T < n_max
    ks = [k for k in ks if k < T.shape[1]]
    vals = []
    for k in ks:
        d = np.abs(T[:, :-k] - T[:, k:])
        w = ok[:, :-k] & ok[:, k:]
        vals.append(float(d[w].mean()) if w.sum() >= 8 else np.nan)
    good = [(k, v) for k, v in zip(ks, vals) if np.isfinite(v) and v > 0]
    med = float(np.median(T[ok])) if bool(ok.any()) else np.nan
    sat = vals[0] / med if (vals and np.isfinite(vals[0]) and med > 0) else np.nan
    H, r2 = np.nan, np.nan
    if len(good) >= 3:
        lk = np.log([g[0] for g in good])
        lv = np.log([g[1] for g in good])
        coef = np.polyfit(lk, lv, 1)
        H = float(coef[0])
        resid = lv - np.polyval(coef, lk)
        r2 = float(1.0 - resid.var() / lv.var()) if lv.var() > 0 else np.nan
    raw = [float(np.abs(T[:, :-k] - T[:, k:]).mean()) for k in ks]
    H_raw = float(np.polyfit(np.log(ks), np.log(raw), 1)[0]) if min(raw) > 0 else np.nan
    # Two independent ways a low H can be meaningless, and each catches cases the other
    # misses.  The saturation ratio flags a variogram that starts flat; R^2 flags one that
    # is not a power law at all.  A field with sat = 0.47 (just inside tolerance) and
    # R^2 = 0.934 returned H = 0.079 and was pure decorrelation.
    if (np.isfinite(sat) and sat > sat_tol) or (np.isfinite(r2) and r2 < r2_min):
        H = np.nan
    return H, H_raw, float(1.0 - ok.mean()), float(sat), float(r2)


def uncertainty_alpha(T, n_max, ks=(1, 2, 4, 8, 16, 32)):
    """Grebogi-McDonald-Ott-Yorke uncertainty exponent, as an independent check on H.

    f(eps) = fraction of pixel pairs at separation eps that land on opposite sides of the
    median.  For a self-affine field of exponent H the level sets have dimension 2 - H, so
    theory says **alpha = H** -- and the two statistics fail in completely different ways
    (the variogram measures the magnitude of |dT|, this counts sign changes across a
    threshold), which makes agreement real corroboration rather than arithmetic restated.

    Measured over 14 fields: Pearson(alpha, H) = +0.90, mean |alpha - H| = 0.12, with
    smooth fields correctly at alpha ~ 0.95-1.00.  alpha runs systematically ~0.1 above H,
    so it is a cross-check and not a substitute.

    Two things this must do or it reproduces the errors that made two earlier exponents in
    this project spurious: binarise (on the multi-valued field, "T differs at all"
    saturates at one pixel), and fit only where `3/npairs <= f < 0.35` -- above the
    sampling floor, below saturation.

    Returns (alpha, R^2, f_of_eps).
    """
    T = np.asarray(T, float)
    ok = T < n_max
    b = T >= np.median(T[ok])
    ks = np.array([k for k in ks if k < T.shape[1] // 2])
    f, npair = [], 1
    for k in ks:
        d = t = 0
        for ax in (0, 1):
            x = np.take(b, np.arange(b.shape[ax] - k), axis=ax)
            y = np.take(b, np.arange(k, b.shape[ax]), axis=ax)
            w = (np.take(ok, np.arange(ok.shape[ax] - k), axis=ax)
                 & np.take(ok, np.arange(k, ok.shape[ax]), axis=ax))
            d += int((x[w] != y[w]).sum())
            t += int(w.sum())
        f.append(d / max(t, 1))
        npair = t
    f = np.array(f)
    m = (f >= 3.0 / npair) & (f < 0.35)
    if m.sum() < 3:
        return np.nan, np.nan, f
    c = np.polyfit(np.log(ks[m]), np.log(f[m]), 1)
    resid = np.log(f[m]) - np.polyval(c, np.log(ks[m]))
    lv = np.log(f[m])
    r2 = float(1.0 - resid.var() / lv.var()) if lv.var() > 0 else np.nan
    return float(c[0]), r2, f


def interface_mask(binary):
    """Pixels of a boolean field that touch the other label (4-neighbourhood)."""
    b = np.asarray(binary, bool)
    m = np.zeros(b.shape, bool)
    m[:, :-1] |= b[:, :-1] != b[:, 1:]
    m[:, 1:] |= b[:, 1:] != b[:, :-1]
    m[:-1, :] |= b[:-1, :] != b[1:, :]
    m[1:, :] |= b[1:, :] != b[:-1, :]
    return m


def box_D(mask, min_count=4):
    """Box dimension of a boolean mask, plus counts and per-octave ratios.

    Box-count BINARY partitions only.  With many labels "differs from a neighbour" is true
    almost everywhere, the set is space-filling and D comes back near 2 even for a
    perfectly smooth field.  Ratio 2.0 per doubling is a smooth curve; a constant ratio
    above 2 is scale-invariant roughness.
    """
    cur, sizes, counts, d = np.asarray(mask, bool), [], [], 1
    while cur.shape[0] >= 4 and cur.shape[1] >= 4:
        counts.append(int(cur.sum()))
        sizes.append(d)
        h, w = (cur.shape[0] // 2) * 2, (cur.shape[1] // 2) * 2
        cur = cur[:h, :w].reshape(h // 2, 2, w // 2, 2).any(axis=(1, 3))
        d *= 2
    ratios = [counts[i] / counts[i + 1] for i in range(len(counts) - 1)
              if counts[i + 1] > 0]
    s, c = np.array(sizes, float), np.array(counts, float)
    ok = c >= min_count
    D_ = float(-np.polyfit(np.log(s[ok]), np.log(c[ok]), 1)[0]) if ok.sum() >= 3 else None
    return D_, counts, ratios


def smooth_null(res, n=5):
    """What box_D returns for boundaries that are exactly smooth, at this resolution.

    Below ~500 px this estimator cannot distinguish 1.0 from 1.3 -- a straight line reads
    1.200 at res 96.  Never quote D without this band.
    """
    y, x = np.mgrid[0:res, 0:res] / (res - 1) * 2 - 1
    shapes = [x > 0.013, (x ** 2 + y ** 2) < 0.55 ** 2,
              y > 0.25 * np.sin(2 * np.pi * x),
              y > 0.4 * np.sin(3 * np.pi * x),
              y > 0.9 * x ** 3 + 0.5 * x - 0.15]
    return [d for d in (box_D(interface_mask(b))[0] for b in shapes[:n]) if d is not None]


def n_components(mask):
    """4-connected components of a boolean mask (iterative flood fill)."""
    m = np.asarray(mask, bool)
    seen = np.zeros(m.shape, bool)
    H_, W_ = m.shape
    n = 0
    for sy in range(H_):
        for sx in range(W_):
            if not m[sy, sx] or seen[sy, sx]:
                continue
            n += 1
            st = [(sy, sx)]
            seen[sy, sx] = True
            while st:
                y, x = st.pop()
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < H_ and 0 <= nx < W_ and m[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True
                        st.append((ny, nx))
    return n


def field_stats(T, n_max, res, null):
    F = np.asarray(T, float).reshape(res, res)
    H, H_raw, cens, sat, r2 = variogram_H(F, n_max)
    alpha, alpha_r2, _ = uncertainty_alpha(F, n_max)
    iface = interface_mask(F >= np.median(F))
    D_, _, ratios = box_D(iface)
    return dict(H=H, H_raw=H_raw, D=D_, ratios=ratios, saturation=sat, r2=r2,
                alpha=alpha, alpha_r2=alpha_r2,
                frac_censored=cens, n_distinct_T=int(len(np.unique(F[F < n_max]))),
                n_components=n_components(iface), smooth_null=null)


def roughest_centre(T, n_max, cx, cy, width, win=16, margin=8, step=4):
    """Centre of the window with the largest mean adjacent |dT|, ignoring censored pixels.

    Zoom-centre selection is the largest single source of wasted renders here.  Gradient
    magnitude sends the sequence into smooth ridges; interface density does too.  On a
    slim fractal -- rough on a thin set, smooth almost everywhere else -- a centre chosen
    any other way lands in the smooth bulk and the sequence "disproves" structure that is
    really there.  The score must also mask censored pixels, or a window sitting in the
    n_max plateau scores as perfectly flat and repels the search from the slowest region.

    Returns (score, cx, cy, max_T_in_window); the last drives the next level's cap.
    """
    T = np.asarray(T, float)
    res = T.shape[0]
    ok = T < n_max
    lx = np.linspace(cx - width, cx + width, res)
    ly = np.linspace(cy - width, cy + width, res)
    best = None
    for iy in range(margin, max(margin + 1, res - win - margin), step):
        for ix in range(margin, max(margin + 1, res - win - margin), step):
            s, w = T[iy:iy + win, ix:ix + win], ok[iy:iy + win, ix:ix + win]
            num = den = 0.0
            for ax in (0, 1):
                d = np.abs(np.diff(s, axis=ax))
                m = (np.take(w, np.arange(s.shape[ax] - 1), axis=ax)
                     & np.take(w, np.arange(1, s.shape[ax]), axis=ax))
                num += float(d[m].sum())
                den += float(m.sum())
            if den < 8:
                continue
            r = num / den
            if best is None or r > best[0]:
                tw = float(n_max) if not bool(w.all()) else float(s[w].max())
                best = (r, lx[ix + win // 2], ly[iy + win // 2], tw)
    return best


# ----------------------------------------------------------------------- chase
def chase(recs, question, levels=4, zoom=10.0, res=128, radius=0.3, eta=1.0, tau=0.05,
          n_max=1500, cap_margin=3.0, n_max_cap=40000, center=(0.0, 0.0), width=1.0,
          chunk=256, seed=0, outdir=None, tag=""):
    """Nested zoom, re-centring on the roughest window at each level.

    STEP SIZE MATTERS.  The rough set is a thin filament and the chase only re-centres
    between levels, so a step too large overshoots it and lands in the smooth bulk -- which
    looks exactly like the roughness resolving away.  The same maze reported H = 1.070 with
    2 distinct settling times at 10^4x using 100x steps, and H = 0.331 with 110 using 10x
    steps.  Keep `zoom` at or below ~10.

    The cap is predicted from the previous level's chosen window rather than discovered by
    retrying, because a retry re-renders everything already computed and the levels that
    need it are the expensive ones.
    """
    for r in recs:
        r.set_question(question)
    base, u, v = plane_basis(seed)
    null = smooth_null(res)
    cx, cy, w = center[0], center[1], width
    rows = []
    t0 = time.time()
    for lvl in range(levels + 1):
        coords = grid((cx, cy), w, res)
        T, _ = settle_field_sharded(recs, base, u, v, coords, radius, eta, n_max, tau,
                                    chunk=chunk, label=f"L{lvl} nmax{n_max} ")
        st = field_stats(T, n_max, res, null)
        got = roughest_centre(T.reshape(res, res), n_max, cx, cy, w) \
            if lvl < levels else None
        row = dict(level=lvl, zoom=width / w, center=[cx, cy], width=w, n_max=n_max,
                   res=res, tau=tau, radius=radius, eta=eta,
                   next_center=[got[1], got[2]] if got else None, **st)
        rows.append(row)
        if outdir:
            os.makedirs(outdir, exist_ok=True)
            np.savez_compressed(os.path.join(outdir, f"chase_{tag}_L{lvl}.npz"),
                                T_int=T.reshape(res, res), meta=json.dumps(row))
            json.dump(rows, open(os.path.join(outdir, f"chase_{tag}.json"), "w"), indent=1)
        Hs = f"{st['H']:.3f}" if np.isfinite(st['H']) else "  NaN (SATURATED)"
        print(f"[L{lvl}] zoom {row['zoom']:7.0f}x  H={Hs}  "
              f"D={st['D']:.3f} null[{min(null):.2f},{max(null):.2f}]  "
              f"distinctT={st['n_distinct_T']}  comps={st['n_components']}  "
              f"cens={100 * st['frac_censored']:.1f}%  sat={st['saturation']:.2f}  "
              f"R2={st['r2']:.3f}  "
              f"[{(time.time() - t0) / 60:.0f} min]", flush=True)
        if got is None:
            break
        cx, cy = got[1], got[2]
        w /= zoom
        n_max = int(min(max(n_max, np.ceil(cap_margin * got[3])), n_max_cap))
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mazes", nargs="+", required=True, help="maze JSON from mazes.py")
    p.add_argument("--hash", required=True, help="8-char maze hash prefix")
    p.add_argument("--devices", nargs="+", default=["cuda:0"])
    p.add_argument("--outdir", default="outputs/chase")
    p.add_argument("--res", type=int, default=128)
    p.add_argument("--levels", type=int, default=4)
    p.add_argument("--zoom", type=float, default=10.0, help="keep at or below ~10")
    p.add_argument("--radius", type=float, default=0.3)
    p.add_argument("--eta", type=float, default=1.0)
    p.add_argument("--tau", type=float, default=0.05)
    p.add_argument("--n-max", type=int, default=1500)
    p.add_argument("--chunk", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    print(f"pid {os.getpid()}", flush=True)

    mz = {}
    for f in a.mazes:
        if os.path.exists(f):
            for m in json.load(open(f)):
                mz.setdefault(m["hash"][:8], m)
    m = mz[a.hash]
    print(f"maze {a.hash}  rating {m['rating']}  tau {a.tau}  R {a.radius}  "
          f"eta {a.eta}  res {a.res}  zoom {a.zoom}x/level", flush=True)
    recs = [Recurrence(m["question"], device=d, dtype=torch.float64) for d in a.devices]
    print(f"sharding each field across {len(recs)} GPU(s): {a.devices}", flush=True)

    rows = chase(recs, m["question"], levels=a.levels, zoom=a.zoom, res=a.res,
                 radius=a.radius, eta=a.eta, tau=a.tau, n_max=a.n_max,
                 chunk=a.chunk, seed=a.seed, outdir=a.outdir, tag=a.hash)

    print("\n" + "=" * 72)
    print(f"{'lvl':>3s} {'zoom':>9s} {'n_max':>7s} {'cens':>6s} {'H':>8s} {'D':>7s} "
          f"{'distinctT':>10s} {'comps':>6s}")
    for r in rows:
        Hs = f"{r['H']:8.3f}" if np.isfinite(r["H"]) else "     NaN"
        print(f"{r['level']:3d} {r['zoom']:8.0f}x {r['n_max']:7d} "
              f"{100 * r['frac_censored']:5.1f}% {Hs} {r['D']:7.3f} "
              f"{r['n_distinct_T']:10d} {r['n_components']:6d}")
    Hs = [r["H"] for r in rows if np.isfinite(r["H"])]
    nT = [r["n_distinct_T"] for r in rows]
    print(f"\nH over {rows[-1]['zoom']:.0f}x: {[round(h, 3) for h in Hs]}")
    print(f"distinct T: {nT}  (rises = new structure resolving; falls = the range "
          f"collapsing, which is the smooth signature)")
    print("VERDICT: " + ("SCALE-INVARIANT roughness -- H stays well below 1 throughout"
                         if Hs and max(Hs) < 0.7 else
                         "H rises toward 1 -- the roughness resolves away"))


if __name__ == "__main__":
    main()
