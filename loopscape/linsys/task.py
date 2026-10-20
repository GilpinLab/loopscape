"""Square, full-rank linear systems ``A x = b`` over GF(3), as fixed-length token sequences::

    a11 .. a1n | b1 ;  a21 .. a2n | b2 ;  ...  ;  =  ? ? .. ?

The ``?`` answer slots are the only supervised positions and are never fed the answer.
"""
from __future__ import annotations

import hashlib

import numpy as np
import torch

P, N = 3, 8
KMAX = 8
TEST_FRAC = 0.05

BAR, ROW, EQ, SLOT, PAD = P, P + 1, P + 2, P + 3, P + 4
SEQ_LEN = N * (N + 3) + 1 + N


def _rank(M, p=P):
    R = M.copy() % p
    rank = 0
    for c in range(R.shape[1]):
        nz = np.nonzero(R[rank:, c])[0]
        if rank >= R.shape[0] or len(nz) == 0:
            continue
        i = rank + nz[0]
        R[[rank, i]] = R[[i, rank]]
        R[rank] = (R[rank] * pow(int(R[rank, c]), p - 2, p)) % p
        for i in range(R.shape[0]):
            if i != rank and R[i, c]:
                R[i] = (R[i] - R[i, c] * R[rank]) % p
        rank += 1
    return rank


def is_test(A, b, test_frac=TEST_FRAC):
    """Deterministic hash split, so train and test instances never overlap."""
    h = hashlib.blake2b(A.astype(np.int8).tobytes() + b"|" + b.astype(np.int8).tobytes(),
                        digest_size=8).digest()
    return (int.from_bytes(h, "big") % 10000) < int(test_frac * 10000)


def _rows(weights, rng, n=N, p=P):
    A = np.zeros((n, n), dtype=np.int64)
    # weight-1 rows go on distinct columns; two on the same column can never be full rank
    ones, heavy = np.where(weights == 1)[0], np.where(weights > 1)[0]
    if len(ones):
        cols = rng.choice(n, size=len(ones), replace=False)
        A[ones, cols] = rng.integers(1, p, size=len(ones))
    for r in heavy:
        cols = rng.choice(n, size=int(weights[r]), replace=False)
        A[r, cols] = rng.integers(1, p, size=len(cols))
    return A if _rank(A, p) == n else None


def sample_instance(rng, split="train", n=N, p=P, kmax=KMAX):
    """``(A, b, x)`` with a unique solution: row weights ``~ U{1..k}``, ``k ~ U{1..kmax}``."""
    while True:
        k = int(rng.integers(1, kmax + 1))
        for _ in range(200):
            A = _rows(rng.integers(1, k + 1, size=n), rng, n, p)
            if A is not None:
                break
        if A is None:
            continue
        x = rng.integers(0, p, size=n)
        b = (A @ x) % p
        if is_test(A, b) == (split == "test"):
            return A, b, x


def encode(A, b):
    ids = []
    for i in range(A.shape[0]):
        ids += [int(v) for v in A[i]] + [BAR, int(b[i]), ROW]
    return ids + [EQ] + [SLOT] * A.shape[1]


def sample_batch(batch_size, rng, split="train"):
    """``inputs`` and ``targets`` (PAD everywhere except the answer slots)."""
    inputs = np.empty((batch_size, SEQ_LEN), dtype=np.int64)
    targets = np.full((batch_size, SEQ_LEN), PAD, dtype=np.int64)
    for i in range(batch_size):
        A, b, x = sample_instance(rng, split)
        inputs[i] = encode(A, b)
        targets[i, -N:] = x
    return torch.from_numpy(inputs), torch.from_numpy(targets)


def answer_code(x, p=P):
    """Integer label of an answer vector (batched along the last axis)."""
    return (np.asarray(x) * p ** np.arange(np.shape(x)[-1])).sum(-1)


def peel_depths(A):
    """Round at which substitution pins each unknown; -1 means it needs elimination."""
    n = A.shape[1]
    depth = -np.ones(n, dtype=int)
    known = np.zeros(n, dtype=bool)
    rnd = 0
    while True:
        rnd += 1
        new = [np.flatnonzero(row & ~known)[0] for row in (A != 0)
               if (row & ~known).sum() == 1]
        if not new:
            return depth
        for j in set(new):
            depth[j], known[j] = rnd, True
