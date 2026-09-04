"""HRM-Text solver -- Hierarchical Reasoning Model on text (arXiv:2605.20613).

HRM-Text-1B (``sapientinc/HRM-Text-1B``) is the HRM architecture -- the same
dual-timescale z_H/z_L recurrence family as EqR -- trained as a prefix-LM on
text.  Two 16-layer transformer stacks iterate over shared latents:

    z_H(0) = embed(tokens) * embedding_scale        # slow / high-level state
    z_L(0) = z_L_init                               # fast / low-level state
    repeat (one "step" = one H-cycle):
        z_L <- L_module(z_L + z_H)   x L_cycles (3)
        z_H <- H_module(z_H + z_L)
    logits = lm_head(z_H)

Unlike Huginn there is **no random init and no per-step noise**: the whole solve
is a deterministic function of the prompt (and any injected latents), so there
is no ``seed`` and ``solve_batch`` has no seeds mode -- batched sweeps take
``z_H``/``z_L`` tensors, mirroring EqR's latent-injection API.  Note also that
the input enters *only* through z_H's initial condition (there is no per-step
embedding injection as in Huginn/EqR): iterating past the trained
``H_cycles = 2`` simply continues the autonomous (z_H, z_L) dynamical system,
which is exactly the off-distribution knob this wrapper exposes via
``num_steps``.

Text semantics match :class:`HuginnSolver`: ``puzzle`` is a prompt string,
``grid`` the greedy token id per position, ``extra["decoded_text"]`` the
next-token continuation.  Two HRM-Text-specific fidelity knobs:

* ``prefix_mask`` (default True): the model was pre-trained with a PrefixLM
  mask -- the prompt attends bidirectionally.  Disabling it falls back to pure
  causal attention, which is off-distribution (noticeably worse logits).
* ``condition``: the checkpoint expects prompts wrapped as
  ``<|im_start|>{condition tokens}{text}<|im_end|>`` with comma-composable
  condition tags (``direct`` | ``cot`` | ``noisy`` | ``synth``).  Pass e.g.
  ``condition="synth,cot"`` (the README's reasoning mode) to wrap the prompt
  automatically, or leave ``None`` to send the prompt verbatim.

The model class is native to transformers (>= 5.9), so nothing is vendored;
weights are downloaded on first use (see :mod:`_hrm_text_download`).
"""
from __future__ import annotations

from typing import List, Optional, Union

import torch

from ..base import BatchSolveResult, Solver, SolveResult, register_solver
from ..huginn.solver import _matches_expected, _merge_batch_results, _residual
from .download import ensure_hrm_text_checkpoint

_LATENT_DIM = 1536   # config.hidden_size; both latents are (B, T, 1536)

# Condition tags -> tokenizer special tokens (from the model card; tags compose
# comma-separated, in order, into a single prefix block).
_CONDITION_TOKENS = {
    "direct": "<|object_ref_start|>",
    "cot": "<|object_ref_end|>",
    "noisy": "<|quad_start|>",
    "synth": "<|quad_end|>",
}


class HRMTextSolver(Solver):
    """HRM on text -- deterministic dual-latent (z_H/z_L) recurrent solver."""

    name = "hrm-text"

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
    ) -> "HRMTextSolver":
        """Download (first use) and load HRM-Text-1B.

        ``dtype`` defaults to bfloat16 (the training/storage precision, ~2.4 GB).
        """
        from transformers import AutoModelForCausalLM, AutoTokenizer

        ckpt_dir = ensure_hrm_text_checkpoint(checkpoint)
        model = AutoModelForCausalLM.from_pretrained(ckpt_dir, dtype=dtype)
        model = model.to(device=device)
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(ckpt_dir)
        return cls(model, tokenizer, device=device)

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def format_prompt(prompt: str, condition: Optional[str] = "synth,cot") -> str:
        """Wrap a bare prompt in the checkpoint's expected envelope,
        ``<|im_start|>{condition tokens}{prompt}<|im_end|>``.  ``condition`` is a
        comma-separated composition of ``direct``/``cot``/``noisy``/``synth``
        (order preserved); ``None`` returns the prompt unchanged."""
        if condition is None:
            return prompt
        try:
            prefix = "".join(_CONDITION_TOKENS[t.strip()] for t in condition.split(","))
        except KeyError as e:
            raise ValueError(f"Unknown condition tag {e.args[0]!r}; "
                             f"expected tags among {sorted(_CONDITION_TOKENS)}") from None
        return f"<|im_start|>{prefix}{prompt}<|im_end|>"

    def _encode(self, prompt: str) -> torch.Tensor:
        device = next(self.model.parameters()).device
        # No BOS is auto-added; the <|im_start|> envelope carries the specials.
        return self.tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)

    def _setup(self, input_ids: torch.Tensor, B: int, prefix_mask: bool):
        """Embed the prompt and build the (Prefix-LM) attention mask + rotary
        tables -- the parts of HrmTextModel.forward that precede the recurrence."""
        from transformers.masking_utils import create_causal_mask

        m = self.model.model                                    # HrmTextModel
        embeds = m.embed_tokens(input_ids) * m.embedding_scale  # (1, T, D)
        embeds = embeds.expand(B, -1, -1).contiguous() if B > 1 else embeds
        T = embeds.shape[1]
        position_ids = torch.arange(T, device=embeds.device).unsqueeze(0).expand(B, -1)

        mask_kwargs = dict(config=m.config, inputs_embeds=embeds, attention_mask=None,
                           past_key_values=None, position_ids=position_ids)
        if prefix_mask and m.config.prefix_lm:
            # token_type_ids == 1 everywhere: the whole prompt is one
            # bidirectional prefix block (the model card's recommended call).
            mask_kwargs["block_sequence_ids"] = torch.zeros(
                (B, T), dtype=torch.long, device=embeds.device)
        attn = create_causal_mask(**mask_kwargs)
        pos_emb = m.rotary_emb(embeds, position_ids)
        return embeds, attn, position_ids, pos_emb

    def _prepare_latent(self, z, T: int, B: int, name: str) -> torch.Tensor:
        device = next(self.model.parameters()).device
        dtype = next(self.model.parameters()).dtype
        t = torch.as_tensor(z, device=device, dtype=dtype)
        if t.ndim == 2:
            t = t.unsqueeze(0)
        if t.shape[0] == 1 and B > 1:
            t = t.expand(B, -1, -1)
        if tuple(t.shape) != (B, T, _LATENT_DIM):
            raise ValueError(
                f"{name} must have shape {(B, T, _LATENT_DIM)} or {(T, _LATENT_DIM)}, "
                f"got {tuple(t.shape)} (T = prompt token count incl. the envelope)"
            )
        return t.contiguous()

    def _h_cycle(self, z_H, z_L, embeds, attn, pos_emb, position_ids,
                 reinject_input: bool):
        """One solver step = one H-cycle: L_cycles low-level updates, then one
        high-level update -- verbatim the loop body of HrmTextModel.forward.

        With ``reinject_input=True`` the scaled prompt embeddings are added into
        every L-update (``z_L <- L(z_L + z_H + embeds)``) -- the input anchoring
        the original grid-HRM/EqR recurrence has and HRM-Text drops.  This is an
        off-distribution modification: it turns the trained autonomous flow into
        an input-conditioned map (whose fixed points differ from the trained
        step-2 read-out state)."""
        m = self.model.model
        anchor = embeds if reinject_input else 0
        for _ in range(m.config.L_cycles):
            z_L = m.L_module(z_L + z_H + anchor, attention_mask=attn,
                             position_embeddings=pos_emb, position_ids=position_ids)
        z_H = m.H_module(z_H + z_L, attention_mask=attn,
                         position_embeddings=pos_emb, position_ids=position_ids)
        return z_H, z_L

    @torch.no_grad()
    def _run_recurrence(self, z_H, z_L, embeds, attn, pos_emb, position_ids, *,
                        num_steps: int, exit_threshold: Optional[float],
                        return_intermediates: bool, return_latents,
                        reinject_input: bool = False, damping: float = 1.0,
                        latent_positions=None, latents_dtype: torch.dtype = torch.float32):
        keep = {True: ("z_H", "z_L"), "both": ("z_H", "z_L"), "z_H": ("z_H",),
                "z_L": ("z_L",), False: (), None: ()}.get(return_latents)
        if keep is None:
            raise ValueError("return_latents must be True/False/'z_H'/'z_L'/'both', "
                             f"got {return_latents!r}")
        pos = (None if latent_positions is None
               else [p % z_H.shape[1] for p in latent_positions])

        def _rec(z):
            z = z.detach()
            if pos is not None:
                z = z[:, pos]
            return z.to(device="cpu", dtype=latents_dtype)

        steps, halted = 0, False
        intermediates = [] if return_intermediates else None
        residual_history: List[List[float]] = []
        z_H_hist = [] if "z_H" in keep else None
        z_L_hist = [] if "z_L" in keep else None

        logits = None
        for _ in range(num_steps):
            z_H_prev, z_L_prev = z_H, z_L
            z_H, z_L = self._h_cycle(z_H, z_L, embeds, attn, pos_emb, position_ids,
                                     reinject_input)
            if damping != 1.0:
                # Krasnoselskii-Mann averaging on the H-cycle map: forces the
                # iteration to settle while preserving the map's fixed points.
                z_H = (1.0 - damping) * z_H_prev + damping * z_H
                z_L = (1.0 - damping) * z_L_prev + damping * z_L
            steps += 1
            res = _residual(z_H, z_H_prev)
            residual_history.append(res.tolist())
            if z_H_hist is not None:
                z_H_hist.append(_rec(z_H))
            if z_L_hist is not None:
                z_L_hist.append(_rec(z_L))
            if return_intermediates:
                logits = self.model.lm_head(z_H).float()
                intermediates.append(logits.argmax(dim=-1).cpu().tolist())
            if exit_threshold is not None and bool((res <= exit_threshold).all()):
                halted = True
                break

        if logits is None or not return_intermediates:
            logits = self.model.lm_head(z_H).float()
        return logits, steps, halted, intermediates, residual_history, z_H_hist, z_L_hist

    def _decode_texts(self, logits: torch.Tensor) -> List[str]:
        next_ids = logits[:, -1].argmax(dim=-1).tolist()
        return [self.tokenizer.decode([int(t)]) for t in next_ids]

    def _initial_latents(self, embeds, z_H, z_L, T: int, B: int):
        """Trained inits (z_H = scaled embeds, z_L = z_L_init) unless injected."""
        m = self.model.model
        if z_H is None:
            z_H0 = embeds
        else:
            z_H0 = self._prepare_latent(z_H, T, B, "z_H")
        if z_L is None:
            z_L0 = m.z_L_init.to(dtype=embeds.dtype, device=embeds.device) \
                             .expand_as(embeds).contiguous()
        else:
            z_L0 = self._prepare_latent(z_L, T, B, "z_L")
        return z_H0, z_L0

    # ------------------------------------------------------------------ solve
    @torch.no_grad()
    def solve(
        self,
        puzzle: Union[str, "List[List[int]]"],
        *,
        num_steps: Optional[int] = None,
        return_intermediates: bool = False,
        return_latents: Union[bool, str] = False,
        latent_positions: Optional[List[int]] = None,
        latents_dtype: torch.dtype = torch.float32,
        z_H: Optional[torch.Tensor] = None,
        z_L: Optional[torch.Tensor] = None,
        condition: Optional[str] = None,
        prefix_mask: bool = True,
        reinject_input: bool = False,
        damping: float = 1.0,
        exit_threshold: Optional[float] = None,
        expected: Optional[str] = None,
        **model_kwargs,
    ) -> SolveResult:
        """Run the HRM-Text recurrence on a text prompt, reading out each H-cycle.

        Parameters
        ----------
        puzzle        : the text prompt (str).  Pass it pre-wrapped in the
                        ``<|im_start|>...<|im_end|>`` envelope, or supply
                        ``condition`` to wrap it here (see :meth:`format_prompt`).
        num_steps     : H-cycles to run.  Default is the trained
                        ``config.H_cycles`` (2); larger values continue the
                        autonomous (z_H, z_L) dynamics off-distribution.
        z_H, z_L      : optional starting latents ``(T, 1536)`` or ``(1, T, 1536)``
                        replacing the trained inits (z_H: scaled embeddings;
                        z_L: the checkpoint's ``z_L_init``).  The solve is fully
                        deterministic either way -- there is no ``seed``.
        condition     : condition tags to wrap the prompt with (e.g. ``"direct"``,
                        ``"synth,cot"``); ``None`` (default) sends it verbatim.
        prefix_mask   : mark the whole prompt as one bidirectional Prefix-LM
                        block (the trained masking; default True).  ``False`` =
                        pure causal, off-distribution.
        reinject_input: add the scaled prompt embeddings into every L-update
                        (``z_L <- L(z_L + z_H + embeds)``), restoring the
                        input anchoring that grid-HRM/EqR/FPRM recurrences have.
                        Off-distribution (default False = trained autonomous
                        dynamics); use it to probe FPRM-style input-conditioned
                        convergence over many steps.
        damping       : Krasnoselskii-Mann averaging factor ``alpha`` in
                        ``z <- (1 - alpha) * z + alpha * step(z)`` applied to
                        both latents after each H-cycle.  ``1.0`` (default) is
                        the raw trained update; ``alpha < 1`` damps the
                        iteration toward settling while preserving the step
                        map's fixed points.  Off-distribution when != 1.
        exit_threshold: stop once the z_H update ``||z_H(t) - z_H(t-1)||`` (L2
                        over hidden, mean over positions) drops below this;
                        sets ``halted=True``.
        expected      : optional expected continuation for the loose ``solved``
                        prefix match (as in HuginnSolver).
        return_latents: record per-step latents in ``extra["z_H_history"]`` /
                        ``extra["z_L_history"]``, CPU, ``(steps, T', 1536)``.
                        ``True``/``"both"`` records both latents, ``"z_H"`` or
                        ``"z_L"`` just one (halves memory; z_H is usually the
                        one analysed).
        latent_positions: token positions to keep in the recorded histories
                        (e.g. ``[-1]`` for just the readout position -- a
                        ``1/T`` memory saving, the big lever for mesh sweeps).
                        ``None`` keeps all T positions.
        latents_dtype : storage dtype for the histories (default float32;
                        ``torch.float16`` halves memory, ample for bf16 values).

        Returns a :class:`SolveResult` where ``grid`` holds the greedy token id
        per position and ``extra["decoded_text"]`` the next-token continuation.
        """
        prompt = puzzle
        if not isinstance(prompt, str):
            raise TypeError("HRMTextSolver expects a text prompt (str) as the puzzle.")
        if not 0.0 < damping <= 1.0:
            raise ValueError("damping must be in (0, 1]")
        prompt = self.format_prompt(prompt, condition)

        input_ids = self._encode(prompt)                              # (1, T)
        embeds, attn, position_ids, pos_emb = self._setup(input_ids, 1, prefix_mask)
        T = input_ids.shape[1]
        z_H0, z_L0 = self._initial_latents(embeds, z_H, z_L, T, 1)

        steps_cap = int(num_steps) if num_steps is not None else int(self.model.config.H_cycles)
        logits, steps, halted, intermediates, residual_history, z_H_hist, z_L_hist = \
            self._run_recurrence(
                z_H0, z_L0, embeds, attn, pos_emb, position_ids,
                num_steps=steps_cap, exit_threshold=exit_threshold,
                return_intermediates=return_intermediates, return_latents=return_latents,
                reinject_input=reinject_input, damping=damping,
                latent_positions=latent_positions, latents_dtype=latents_dtype,
            )

        decoded_text = self._decode_texts(logits)[0]
        residuals = [r[0] for r in residual_history]
        extra = {
            "decoded_text": decoded_text,
            "prompt_tokens": T,
            "num_steps": steps_cap,
            "condition": condition,
            "prefix_mask": prefix_mask,
            "reinject_input": reinject_input,
            "damping": damping,
        }
        if return_intermediates:
            extra["intermediate_texts"] = [
                self.tokenizer.decode([ids[0][-1]]) for ids in intermediates
            ]
        if z_H_hist:
            extra["z_H_history"] = torch.cat(z_H_hist)                # (steps, T', 1536)
        if z_L_hist:
            extra["z_L_history"] = torch.cat(z_L_hist)

        return SolveResult(
            grid=logits.argmax(dim=-1).squeeze(0).cpu().tolist(),
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
        z_H: Optional[torch.Tensor] = None,
        z_L: Optional[torch.Tensor] = None,
        num_steps: Optional[int] = None,
        return_intermediates: bool = False,
        return_latents: Union[bool, str] = False,
        latent_positions: Optional[List[int]] = None,
        latents_dtype: torch.dtype = torch.float32,
        condition: Optional[str] = None,
        prefix_mask: bool = True,
        reinject_input: bool = False,
        damping: float = 1.0,
        exit_threshold: Optional[float] = None,
        expected: Optional[str] = None,
        chunk_size: Optional[int] = None,
        **model_kwargs,
    ) -> BatchSolveResult:
        """Run the same prompt from B injected latent conditions in one batch.

        ``z_H`` / ``z_L`` are batched starting latents of shape ``(B, T, 1536)``;
        either may be given (the other keeps its trained init, broadcast over B).
        HRM-Text is deterministic, so there is no seeds mode -- at least one
        latent must be supplied.  ``chunk_size`` bounds peak memory exactly as in
        :meth:`HuginnSolver.solve_batch` (results equivalent up to
        batch-size-dependent kernel rounding).

        ``intermediates`` are ``[steps][B]`` id lists and ``residual_history``
        ``[steps][B]`` z_H update norms, so ``np.swapaxes(np.array(...), 0, 1)``
        gives the usual per-condition layout.  With ``return_latents``,
        ``extra["z_H_history"]`` / ``extra["z_L_history"]`` are
        ``(steps, B, T', 1536)`` CPU tensors.

        **Memory**: chunking bounds the per-forward working set, but recorded
        histories accumulate across chunks: full-resolution latents cost
        ``2 * steps * B * T * 1536 * 4`` bytes of CPU RAM regardless of
        ``chunk_size``.  For large sweeps, shrink what is *recorded*:
        ``return_latents="z_H"`` (2x), ``latent_positions=[-1]`` (T x -- the
        big one; the readout position is usually all you analyse), and
        ``latents_dtype=torch.float16`` (2x).
        """
        prompt = puzzle
        if not isinstance(prompt, str):
            raise TypeError("HRMTextSolver expects a text prompt (str) as the puzzle.")
        if z_H is None and z_L is None:
            raise ValueError(
                "Provide z_H and/or z_L of shape (B, T, 1536): HRM-Text is "
                "deterministic, so a batch of identical trained inits would be "
                "B copies of the same trajectory."
            )

        ref = torch.as_tensor(z_H if z_H is not None else z_L)
        B = ref.shape[0] if ref.ndim == 3 else 1

        # -- optional chunking over the batch dim to bound peak working memory --
        if chunk_size is not None:
            if chunk_size < 1:
                raise ValueError("chunk_size must be >= 1")
            if chunk_size < B:
                def sl(z, s):
                    if z is None:
                        return None
                    t = torch.as_tensor(z)
                    return t[s:s + chunk_size] if t.ndim == 3 else t
                parts = [
                    self.solve_batch(
                        prompt, z_H=sl(z_H, s), z_L=sl(z_L, s), num_steps=num_steps,
                        return_intermediates=return_intermediates,
                        return_latents=return_latents,
                        latent_positions=latent_positions, latents_dtype=latents_dtype,
                        condition=condition,
                        prefix_mask=prefix_mask, reinject_input=reinject_input,
                        damping=damping, exit_threshold=exit_threshold,
                        expected=expected, chunk_size=None, **model_kwargs,
                    )
                    for s in range(0, B, chunk_size)
                ]
                return _merge_batch_results(parts, return_intermediates, return_latents)

        if not 0.0 < damping <= 1.0:
            raise ValueError("damping must be in (0, 1]")
        prompt = self.format_prompt(prompt, condition)
        input_ids = self._encode(prompt)                              # (1, T)
        T = input_ids.shape[1]
        embeds, attn, position_ids, pos_emb = self._setup(input_ids, B, prefix_mask)
        z_H0, z_L0 = self._initial_latents(embeds, z_H, z_L, T, B)

        steps_cap = int(num_steps) if num_steps is not None else int(self.model.config.H_cycles)
        logits, steps, halted, intermediates, residual_history, z_H_hist, z_L_hist = \
            self._run_recurrence(
                z_H0, z_L0, embeds, attn, pos_emb, position_ids,
                num_steps=steps_cap, exit_threshold=exit_threshold,
                return_intermediates=return_intermediates, return_latents=return_latents,
                reinject_input=reinject_input, damping=damping,
                latent_positions=latent_positions, latents_dtype=latents_dtype,
            )

        decoded_texts = self._decode_texts(logits)
        extra = {
            "decoded_texts": decoded_texts,
            "prompt_tokens": T,
            "num_steps": steps_cap,
            "condition": condition,
            "prefix_mask": prefix_mask,
            "reinject_input": reinject_input,
            "damping": damping,
        }
        if z_H_hist:
            extra["z_H_history"] = torch.stack(z_H_hist)              # (steps, B, T', 1536)
        if z_L_hist:
            extra["z_L_history"] = torch.stack(z_L_hist)

        return BatchSolveResult(
            grids=logits.argmax(dim=-1).cpu().tolist(),
            steps=steps,
            solved=[_matches_expected(t, expected) for t in decoded_texts],
            halted=halted,
            solver=self.name,
            intermediates=intermediates,
            residual_history=residual_history,
            extra=extra,
        )


register_solver("hrm-text", HRMTextSolver)
