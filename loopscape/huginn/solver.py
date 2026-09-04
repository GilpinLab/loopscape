"""Huginn solver -- latent recurrent-depth language model (arXiv:2502.05171).

Huginn (``tomg-group-umd/huginn-0125``, 3.5B params) is a *looped* text LM: a
2-layer prelude embeds tokens into a latent space, a 4-layer recurrent core is
iterated ``num_steps`` times from a random initial state (trunc-normal, like
FPRM/EqR's latent inits), and a 2-layer coda + tied LM head decode the latent
into next-token logits.  One recurrence step updates the whole-sequence latent
``s_i = R(e, s_{i-1})`` of shape ``(B, T, 5280)``.

Unlike FPRM/EqR this is a *language* model, so the unified API is interpreted as:

* ``puzzle``  -> a text prompt (str);
* ``grid``    -> the greedy argmax token id per position (a flat ``List[int]``,
  length T) -- the model's decoded read-out, with the next-token continuation
  as text in ``extra["decoded_text"]``;
* ``solved``  -> only meaningful when an ``expected`` continuation is supplied
  (loose prefix match on the greedy next token); ``False`` otherwise;
* ``halted``  -> fires when ``exit_threshold`` is given and the latent update
  norm drops below it (Huginn's "latent-diff" adaptive-compute criterion);
  with the default fixed-step loop it is always ``False``.

The recurrence hooks are the checkpoint's own alternative interfaces
(``embed_inputs`` -> ``initialize_state``/injected latent -> ``iterate_one_step``
-> ``predict_from_latents``), i.e. exactly the upstream adaptive-compute path,
so per-step read-outs match the model's ``forward(num_steps=...)``.

The inference code is vendored (see ``loopscape/huginn/_vendor``) instead of
using ``trust_remote_code``: the upstream files target transformers 4.47 and
need a small compatibility patch under transformers 5.x, applied here at import
time.  Weights (~15.6 GB) are downloaded from the Hub on first use (see
:mod:`_huginn_download`).
"""
from __future__ import annotations

import os
from typing import List, Optional, Union

import torch

from ..base import BatchSolveResult, Solver, SolveResult, register_solver
from .download import ensure_huginn_checkpoint

_LATENT_DIM = 5280   # config.n_embd; the recurrent state is (B, T, 5280)


def _import_huginn_model():
    """Import the vendored RavenForCausalLM, patched for transformers >= 5.

    transformers 5.x changed ``_tied_weights_keys`` from a list of tied param
    names to a ``{target: source}`` dict; the upstream file still ships the 4.x
    list form, which crashes ``tie_weights()`` during ``__init__``.  Patch the
    class attribute before any instantiation.
    """
    from ._vendor.raven_modeling_minimal import RavenForCausalLM

    if isinstance(RavenForCausalLM._tied_weights_keys, (list, tuple)):
        RavenForCausalLM._tied_weights_keys = {"lm_head.weight": "transformer.wte.weight"}
    return RavenForCausalLM


def _residual(x_new: torch.Tensor, x_old: torch.Tensor) -> torch.Tensor:
    """Per-sample latent update ||s_k - s_{k-1}||: L2 over the hidden dim, mean
    over positions -- the same convergence diagnostic as the parcae probe."""
    return (x_new.float() - x_old.float()).norm(dim=-1).mean(dim=-1)


def _matches_expected(decoded_text: str, expected: Optional[str]) -> bool:
    """Loose correctness check for text: the greedy next token and the expected
    continuation agree on their common prefix (whitespace-insensitive)."""
    if expected is None:
        return False
    a, b = decoded_text.strip(), expected.strip()
    return bool(a) and (b.startswith(a) or a.startswith(b))


def _merge_batch_results(parts: List[BatchSolveResult], return_intermediates: bool,
                         return_latents: bool) -> BatchSolveResult:
    """Concatenate per-chunk BatchSolveResults back into one, as if unchunked.

    Chunks can only differ in step count when ``exit_threshold`` made one halt
    early; shorter chunks are padded by repeating their last step (the latent
    has converged there by definition), matching EqR's seed-sweep merging.
    """
    steps = max(p.steps for p in parts)

    def pad(seq: list, chunk_steps: int, i: int):
        return seq[i] if i < chunk_steps else seq[-1]

    intermediates = None
    if return_intermediates:
        intermediates = [
            [row for p in parts for row in pad(p.intermediates, p.steps, i)]
            for i in range(steps)
        ]
    residual_history = [
        [r for p in parts for r in pad(p.residual_history, p.steps, i)]
        for i in range(steps)
    ]

    # Merge extras: per-condition lists concatenate; per-step latent-history
    # tensors -- (steps, B_chunk, ...), e.g. Huginn's ``latent_history`` or
    # HRM-Text's ``z_H_history``/``z_L_history`` -- pad along steps by repeating
    # the last frame, then concatenate along the batch dim; scalar metadata
    # (prompt_tokens, num_steps, ...) is chunk-independent and copied through.
    extra = {}
    for k, v in parts[0].extra.items():
        if k in ("decoded_texts", "seeds"):
            extra[k] = [x for p in parts for x in p.extra[k]]
        elif isinstance(v, torch.Tensor) and v.ndim >= 2:
            padded = [
                torch.cat([h, h[-1:].expand(steps - h.shape[0], *h.shape[1:])])
                if h.shape[0] < steps else h
                for h in (p.extra[k] for p in parts)
            ]
            extra[k] = torch.cat(padded, dim=1)
        else:
            extra[k] = v

    return BatchSolveResult(
        grids=[g for p in parts for g in p.grids],
        steps=steps,
        solved=[s for p in parts for s in p.solved],
        halted=all(p.halted for p in parts),
        solver=parts[0].solver,
        intermediates=intermediates,
        residual_history=residual_history,
        extra=extra,
    )


class HuginnSolver(Solver):
    """Huginn recurrent-depth LM -- deterministic recurrence, random latent init."""

    name = "huginn"

    def __init__(self, model, tokenizer, device: str = "cpu"):
        super().__init__(model, device=device)
        self.tokenizer = tokenizer

    # ------------------------------------------------------------------ loading
    @classmethod
    def load(
        cls,
        checkpoint: Optional[str] = None,
        device: str = "cpu",
        dtype: torch.dtype = torch.bfloat16,
        **kwargs,
    ) -> "HuginnSolver":
        """Download (first use) and load Huginn-0125.

        ``dtype`` defaults to bfloat16, the precision the model was trained and
        benchmarked in (also halves the fp32 checkpoint's memory: ~7 GB).
        ``device="mps"`` works -- the vendored code uses plain SDPA and
        real-valued rotary embeddings.
        """
        RavenForCausalLM = _import_huginn_model()
        from transformers import AutoTokenizer

        ckpt_dir = ensure_huginn_checkpoint(checkpoint)
        model = RavenForCausalLM.from_pretrained(ckpt_dir, dtype=dtype)
        model = model.to(device=device)
        model.eval()
        # trust_remote_code=False: the tokenizer is a standard fast BPE; without
        # the explicit False, the config's auto_map triggers an interactive
        # "run custom code?" prompt (the modeling code is vendored anyway).
        tokenizer = AutoTokenizer.from_pretrained(ckpt_dir, trust_remote_code=False)
        return cls(model, tokenizer, device=device)

    # ------------------------------------------------------------------ helpers
    def _encode(self, prompt: str) -> torch.Tensor:
        device = next(self.model.parameters()).device
        return self.tokenizer.encode(
            prompt, return_tensors="pt", add_special_tokens=True
        ).to(device)

    def _prepare_latent(self, z, T: int, B: int, name: str) -> torch.Tensor:
        device = next(self.model.parameters()).device
        dtype = next(self.model.parameters()).dtype
        t = torch.as_tensor(z, device=device, dtype=dtype)
        if t.ndim == 2:
            t = t.unsqueeze(0)
        if tuple(t.shape) != (B, T, _LATENT_DIM):
            raise ValueError(
                f"{name} must have shape {(B, T, _LATENT_DIM)} or {(T, _LATENT_DIM)}, "
                f"got {tuple(t.shape)} (T = prompt token count incl. BOS)"
            )
        return t

    def _seeded_inits(self, input_embeds: torch.Tensor, seeds, init_scale: float) -> torch.Tensor:
        """One trunc-normal initial latent per seed, stacked to (B, T, 5280)."""
        rows = []
        for s in seeds:
            if s is not None:
                torch.manual_seed(int(s))
            rows.append(self.model.initialize_state(input_embeds[:1], scale=init_scale))
        return torch.cat(rows, dim=0)

    @torch.no_grad()
    def _run_recurrence(
        self,
        input_embeds: torch.Tensor,
        initial_latents: torch.Tensor,
        *,
        num_steps: int,
        exit_threshold: Optional[float],
        return_intermediates: bool,
        return_latents: bool,
        latent_positions=None,
        latents_dtype: torch.dtype = torch.float32,
    ):
        """Shared recurrence loop: iterate the core, reading out each step.

        Returns (final logits, steps run, halted, per-step decoded-id lists,
        per-step residual lists, per-step latent history).
        """
        pos = (None if latent_positions is None
               else [p % initial_latents.shape[1] for p in latent_positions])
        x = initial_latents
        steps, halted = 0, False
        intermediates: Optional[List[List[List[int]]]] = [] if return_intermediates else None
        residual_history: List[List[float]] = []
        latent_history = [] if return_latents else None

        logits = None
        for k in range(num_steps):
            xk = x
            x, _, _ = self.model.iterate_one_step(input_embeds, x, current_step=k)
            steps += 1
            res = _residual(x, xk)
            residual_history.append(res.tolist())
            if return_latents:
                xr = x.detach()
                if pos is not None:
                    xr = xr[:, pos]
                latent_history.append(xr.to(device="cpu", dtype=latents_dtype))
            if return_intermediates:
                logits = self.model.predict_from_latents(x).logits
                intermediates.append(logits.argmax(dim=-1).cpu().tolist())
            if exit_threshold is not None and bool((res <= exit_threshold).all()):
                halted = True
                break

        if logits is None or not return_intermediates:
            logits = self.model.predict_from_latents(x).logits
        return logits, steps, halted, intermediates, residual_history, latent_history

    def _decode_texts(self, logits: torch.Tensor) -> List[str]:
        """Greedy next-token (last position) per batch row, decoded to text."""
        next_ids = logits[:, -1].argmax(dim=-1).tolist()
        return [self.tokenizer.decode([int(t)]) for t in next_ids]

    # ------------------------------------------------------------------ solve
    @torch.no_grad()
    def solve(
        self,
        puzzle: Union[str, "List[List[int]]"],
        *,
        num_steps: int = 32,
        return_intermediates: bool = False,
        return_latents: bool = False,
        seed: Optional[int] = 0,
        initial_latent: Optional[torch.Tensor] = None,
        init_scale: float = 1.0,
        latent_positions: Optional[List[int]] = None,
        latents_dtype: torch.dtype = torch.float32,
        exit_threshold: Optional[float] = None,
        expected: Optional[str] = None,
        **model_kwargs,
    ) -> SolveResult:
        """Run the Huginn recurrence on a text prompt and read out each loop.

        Parameters
        ----------
        puzzle        : the text prompt (str).
        num_steps     : recurrence iterations (paper: coarse below 4, saturates
                        ~64; trained mean is 32).
        seed          : seeds torch's RNG before the random latent init
                        (default 0 for reproducibility; ``None`` leaves the RNG
                        alone).  Ignored when ``initial_latent`` is given --
                        the recurrence itself is deterministic.
        initial_latent: optional starting latent ``(T, 5280)`` or ``(1, T, 5280)``
                        (T = prompt token count incl. BOS) instead of the
                        trunc-normal init -- Huginn's ``input_states``.
        init_scale    : scale of the random init's std (upstream ``init_scale``).
        exit_threshold: if set, stop as soon as the latent update
                        ``||s_k - s_{k-1}||`` (L2 over hidden, mean over
                        positions) drops below it -- the "latent-diff" adaptive
                        criterion; sets ``halted=True``.
        expected      : optional expected continuation; ``solved`` becomes a
                        loose prefix match between it and the greedy next token.
        return_latents: record per-step latents in ``extra["latent_history"]``
                        as a float32 CPU tensor ``(steps, T, 5280)``
                        (~21 kB/token/step).

        Returns a :class:`SolveResult` where ``grid`` holds the greedy token id
        per position and ``extra["decoded_text"]`` the next-token continuation.
        """
        prompt = puzzle
        if not isinstance(prompt, str):
            raise TypeError("HuginnSolver expects a text prompt (str) as the puzzle.")

        input_ids = self._encode(prompt)                             # (1, T)
        input_embeds, _ = self.model.embed_inputs(input_ids)

        if initial_latent is not None:
            x0 = self._prepare_latent(initial_latent, input_embeds.shape[1], 1, "initial_latent")
        else:
            if seed is not None:
                torch.manual_seed(seed)
            x0 = self.model.initialize_state(input_embeds, scale=init_scale)

        logits, steps, halted, intermediates, residual_history, latent_history = \
            self._run_recurrence(
                input_embeds, x0, num_steps=num_steps, exit_threshold=exit_threshold,
                return_intermediates=return_intermediates, return_latents=return_latents,
                latent_positions=latent_positions, latents_dtype=latents_dtype,
            )

        decoded_text = self._decode_texts(logits)[0]
        residuals = [r[0] for r in residual_history]
        extra = {
            "decoded_text": decoded_text,
            "prompt_tokens": int(input_ids.shape[1]),
            "seed": None if initial_latent is not None else seed,
            "num_steps": num_steps,
        }
        if return_intermediates:
            extra["intermediate_texts"] = [
                self.tokenizer.decode([ids[0][-1]]) for ids in intermediates
            ]
        if return_latents:
            extra["latent_history"] = torch.cat(latent_history)     # (steps, T, 5280)

        return SolveResult(
            grid=logits.argmax(dim=-1).squeeze(0).cpu().tolist(),   # greedy ids per position
            steps=steps,
            solved=_matches_expected(decoded_text, expected),
            halted=halted,
            solver=self.name,
            residual=residuals[-1] if residuals else None,
            opt_time=None,
            intermediates=[ids[0] for ids in intermediates] if return_intermediates else None,
            residuals=residuals,
            extra=extra,
        )

    # ------------------------------------------------------------------ batch
    @torch.no_grad()
    def solve_batch(
        self,
        puzzle: Union[str, "List[List[int]]"],
        *,
        initial_latents: Optional[torch.Tensor] = None,
        seeds=None,
        num_steps: int = 32,
        return_intermediates: bool = False,
        return_latents: bool = False,
        init_scale: float = 1.0,
        latent_positions: Optional[List[int]] = None,
        latents_dtype: torch.dtype = torch.float32,
        exit_threshold: Optional[float] = None,
        expected: Optional[str] = None,
        chunk_size: Optional[int] = None,
        **model_kwargs,
    ) -> BatchSolveResult:
        """Run the same prompt from B initial latent conditions in one batch.

        Two ways to specify the B conditions (choose one):

        * ``initial_latents`` : tensor ``(B, T, 5280)`` of starting latents
          (T = prompt token count incl. BOS) -- matches FPRM's batched API.
        * ``seeds``           : list of B RNG seeds; each seeds Huginn's own
          trunc-normal ``initialize_state``.  Because the recurrence itself is
          deterministic (unlike EqR), the seeds only pick the starting point,
          so all B still run in a **single batched forward**.

        ``chunk_size`` : if set, process the B conditions in chunks of at most
        this size and merge -- bounding peak working memory (the transient
        per-step logits are ``B * T * 65536`` floats, the usual MPS OOM cause)
        while returning the same result as one big batch.  ``None`` (default)
        keeps the single-forward behaviour.  Rows never interact, so results
        are equivalent either way up to batch-size-dependent kernel rounding
        (bf16 GEMMs reduce in a different order per batch size; expect ~1e-2
        relative wiggle in residuals and rare argmax flips at early,
        far-from-converged steps).  The one caveat is ``exit_threshold``, which
        upstream halts when *all* rows converge -- under chunking each chunk
        halts on its own, so early chunks may run fewer steps (their trailing
        intermediates/residuals are padded by repeating the last step, and
        ``steps``/``halted`` report the max/AND over chunks).

        ``intermediates`` come back shaped ``[steps][B]`` (each entry a length-T
        id list), so ``np.swapaxes(np.array(result.intermediates), 0, 1)`` gives
        ``(B, steps, T)`` like the other solvers.  ``residual_history`` is
        ``[steps][B]`` latent update norms.  With ``return_latents=True``,
        ``extra["latent_history"]`` is ``(steps, B, T, 5280)`` float32 on CPU
        (large: ~21 kB * B * T per step).
        """
        prompt = puzzle
        if not isinstance(prompt, str):
            raise TypeError("HuginnSolver expects a text prompt (str) as the puzzle.")
        if (initial_latents is None) == (seeds is None):
            raise ValueError("Provide exactly one of initial_latents=(B,T,5280) or seeds=[...].")

        # -- optional chunking over the batch dim to bound peak working memory --
        if chunk_size is not None:
            if chunk_size < 1:
                raise ValueError("chunk_size must be >= 1")
            B_total = (len(list(seeds)) if seeds is not None
                       else (torch.as_tensor(initial_latents).shape[0]
                             if torch.as_tensor(initial_latents).ndim == 3 else 1))
            if chunk_size < B_total:
                if seeds is not None:
                    seeds = list(seeds)
                parts = [
                    self.solve_batch(
                        prompt,
                        initial_latents=(None if initial_latents is None
                                         else torch.as_tensor(initial_latents)[s:s + chunk_size]),
                        seeds=(None if seeds is None else seeds[s:s + chunk_size]),
                        num_steps=num_steps,
                        return_intermediates=return_intermediates,
                        return_latents=return_latents,
                        init_scale=init_scale,
                        latent_positions=latent_positions,
                        latents_dtype=latents_dtype,
                        exit_threshold=exit_threshold,
                        expected=expected,
                        chunk_size=None,
                        **model_kwargs,
                    )
                    for s in range(0, B_total, chunk_size)
                ]
                return _merge_batch_results(parts, return_intermediates, return_latents)

        input_ids = self._encode(prompt)                             # (1, T)
        input_embeds1, _ = self.model.embed_inputs(input_ids)
        T = input_embeds1.shape[1]

        if initial_latents is not None:
            ref = torch.as_tensor(initial_latents)
            B = ref.shape[0] if ref.ndim == 3 else 1
            x0 = self._prepare_latent(initial_latents, T, B, "initial_latents")
        else:
            seeds = list(seeds)
            B = len(seeds)
            x0 = self._seeded_inits(input_embeds1, seeds, init_scale)

        # The prompt is identical across conditions: expand the prelude output
        # instead of re-running it B times.
        input_embeds = input_embeds1.expand(B, -1, -1).contiguous()

        logits, steps, halted, intermediates, residual_history, latent_history = \
            self._run_recurrence(
                input_embeds, x0, num_steps=num_steps, exit_threshold=exit_threshold,
                return_intermediates=return_intermediates, return_latents=return_latents,
                latent_positions=latent_positions, latents_dtype=latents_dtype,
            )

        decoded_texts = self._decode_texts(logits)
        extra = {
            "decoded_texts": decoded_texts,
            "prompt_tokens": int(input_ids.shape[1]),
            "num_steps": num_steps,
        }
        if seeds is not None:
            extra["seeds"] = seeds
        if return_latents:
            extra["latent_history"] = torch.stack(latent_history)   # (steps, B, T, 5280)

        return BatchSolveResult(
            grids=logits.argmax(dim=-1).cpu().tolist(),              # B lists of greedy ids
            steps=steps,
            solved=[_matches_expected(t, expected) for t in decoded_texts],
            halted=halted,
            solver=self.name,
            intermediates=intermediates,
            residual_history=residual_history,
            extra=extra,
        )


register_solver("huginn", HuginnSolver)
