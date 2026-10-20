"""Minimal harness for probing the Parcae looped LM's recurrence, loop by loop.

Parcae (https://arxiv.org/abs/2604.12946) is a *looped* language model: a prelude
maps tokens to an initial latent state, a recurrent "core" block updates that latent
some number of times, then a coda + LM head decode the latent into token logits.

This module lets you run the recurrence for M loops on a prompt and read out, after
*each* loop, both the raw latent state and the answer decoded from it -- so you can
watch the prediction refine as the model "thinks" longer. You can also seed the
recurrence with your own initial latent to control the fixed-point search's start.

Usage
-----
    from loopscape.parcae.probe import load_parcae, loop_trace

    model, tok = load_parcae("SandyResearch/parcae-140m")

NOTE: this project's canonical Parcae is the Countdown-SFT checkpoint
(GilpinLab/parcae-countdown-v1) loaded via :mod:`loopscape.parcae.countdown` --
use that for analyses; the stock SandyResearch loader here is kept for reference.
    trace = loop_trace(model, tok, "The capital of France is", M=8)
    for step in trace:
        print(step.loop, repr(step.decoded_text))

    # control the recurrence initialization:
    import torch
    z0 = torch.zeros(1, seq_len, model.config.recurrent_embedding_dimension)
    trace = loop_trace(model, tok, prompt, M=8, initial_latent=z0)

The API style mirrors the loopscape solvers: a small dataclass result and a load() helper.
Parcae is the odd one out among the loopscape models: it has no ``Solver``
adapter; this probe puts the pinned upstream source on ``sys.path`` directly.
"""

from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor

# NOTE: Parcae checkpoints are large (~5 GB for 1.3b) and are fetched through
# huggingface_hub inside parcae_lm.from_pretrained -- set HF_HOME if you want
# them on a bigger disk than the default cache.

# --- make the pinned upstream parcae source importable -------------------------
# The snapshot is downloaded on first use (see loopscape.parcae.download);
# LOOPSCAPE_PARCAE_SRC overrides it with a local checkout.
from .download import ensure_parcae_source  # noqa: E402

_PARCAE_ROOT = Path(ensure_parcae_source())
if str(_PARCAE_ROOT) not in sys.path:
    sys.path.insert(0, str(_PARCAE_ROOT))

import parcae_lm  # noqa: E402
from parcae_lm.tokenizer import Tokenizer  # noqa: E402


# ------------------------------------------------------------------------------
# Result container (mirrors the SolveResult style used by the loopscape solvers)
# ------------------------------------------------------------------------------
@dataclass
class LoopStep:
    """State read out after one recurrence loop."""

    loop: int  # 1-based loop index
    latent: Tensor  # recurrent state after this loop: (B, T, d_rec), on CPU
    logits: Tensor  # next-token logits decoded from this latent: (B, T, vocab)
    decoded_ids: list[int]  # greedy argmax token id per position (batch item 0)
    decoded_text: str  # greedy continuation token (last position) decoded to text
    residual: float  # ||z_k - z_{k-1}|| mean, a convergence diagnostic
    extra: dict = field(default_factory=dict)


def load_parcae(
    repo_id: str = "SandyResearch/parcae-140m",
    *,
    device: str = "cpu",
    dtype: torch.dtype | None = None,
    tokenizer_repo: str = "SandyResearch/parcae-tokenizer",
):
    """Load a pretrained Parcae model and its tokenizer, ready for probing.

    Parameters
    ----------
    repo_id : HuggingFace repo, e.g. "SandyResearch/parcae-140m" (or -370m/-770m/-1.3b).
    device, dtype : forwarded to the model.
    tokenizer_repo : the shared Parcae BPE tokenizer.
    """
    model = parcae_lm.from_pretrained(repo_id, device=device, dtype=dtype)
    model.eval()
    tok = Tokenizer.from_pretrained(tokenizer_repo)
    return model, tok


# ------------------------------------------------------------------------------
# Core: replicate the Parcae forward flow but stop after each recurrence loop
# and decode the latent, so we can see the answer improve.
# ------------------------------------------------------------------------------
def _run_prelude(model, input_ids: Tensor) -> tuple[Tensor, Tensor]:
    """Tokens -> prelude output embeddings (the recurrence's `input_embeds`)."""
    freqs_cis = model.freqs_cis[:, : input_ids.shape[1]]

    input_embeds = model.transformer.wte(input_ids)
    if model.emb_scale != 1:
        input_embeds = input_embeds * model.emb_scale

    # value-embeddings are keyed off the current input ids (matches Parcae.forward)
    model._current_input_ids = input_ids

    for i, block in enumerate(model.transformer.prelude):
        ve = (
            model.value_embeds[str(i)](input_ids)
            if str(i) in model.value_embeds
            else None
        )
        input_embeds = block(input_embeds, freqs_cis, None, ve=ve)

    if model.config.prelude_norm:
        input_embeds = model.transformer.ln_prelude(input_embeds)
    return input_embeds, freqs_cis


def _decode_latent(model, x: Tensor, input_ids: Tensor, freqs_cis: Tensor) -> Tensor:
    """Latent state -> next-token logits, via C -> coda -> ln_f -> lm_head.

    This is exactly the post-recurrence tail of Parcae.forward (no-labels branch).
    """
    x = model.transformer.C(x)

    coda_ve_offset = (
        model.config.n_layers_in_prelude + model.config.n_layers_in_recurrent_block
    )
    for i, block in enumerate(model.transformer.coda):
        ve_idx = str(coda_ve_offset + i)
        ve = (
            model.value_embeds[ve_idx](input_ids)
            if ve_idx in model.value_embeds
            else None
        )
        x = block(x, freqs_cis, None, ve=ve)
    x = model.transformer.ln_f(x)

    scale = model.config.init.logit_scale
    if model.config.use_fused_head == "full-triton":
        w = (
            model.lm_head.weight.T
            if model.config.tie_embeddings
            else model.lm_head.weight
        )
        logits = torch.matmul(x, w).float() * scale
    else:
        logits = model.lm_head(x).float() * scale

    if model.config.logit_softcap is not None:
        sc = model.config.logit_softcap
        logits = sc * torch.tanh(logits / sc)
    return logits


@contextmanager
def _scaled_dt(model, dt_scale: float):
    """Temporarily rescale the recurrence's input-injection step size ``dt``.

    Parcae's diagonal SSM injection updates the latent as
    ``x_{t+1} = exp(-dt*A) * x_t + dt * (e @ B.T)`` where ``dt = softplus(dt_bias)``
    is a *learned* per-dimension step size (in-distribution value == 1x).

    Scaling ``dt`` by ``dt_scale`` does two things at once, both of which slow
    convergence when ``dt_scale < 1``:
      * shrinks the input-injection gain (``dt * ...``), so each loop moves less, and
      * pushes the state-retention ``exp(-dt*A)`` toward 1, so the latent forgets its
        previous value more slowly -- i.e. the recurrence "loops more" before settling.

    ``dt_scale > 1`` accelerates convergence. This is an off-distribution perturbation
    (the fixed points shift), intended for probing basin/convergence dynamics. We invert
    softplus so the effective step becomes exactly ``dt_scale * dt`` and restore the
    original ``dt_bias`` on exit. A no-op (and a warning) if the adapter has no ``dt_bias``
    (e.g. linear/additive injection) or if ``dt_scale == 1``.
    """
    adapter = model.transformer.adapter
    if dt_scale == 1.0 or not hasattr(adapter, "dt_bias"):
        if dt_scale != 1.0:
            import warnings

            warnings.warn(
                f"dt_scale={dt_scale} ignored: adapter {type(adapter).__name__} has no "
                "'dt_bias' (only 'diagonal' injection supports dt scaling).",
                stacklevel=2,
            )
        yield
        return

    original = adapter.dt_bias.data.clone()
    try:
        # dt_bias' = softplus^{-1}(dt_scale * softplus(dt_bias)) = log(expm1(dt_scale*dt))
        dt = F.softplus(adapter.dt_bias.data)
        adapter.dt_bias.data = torch.log(torch.expm1(dt_scale * dt).clamp_min(1e-6))
        yield
    finally:
        adapter.dt_bias.data = original


@torch.no_grad()
def loop_trace(
    model,
    tokenizer,
    prompt: str,
    M: int,
    *,
    initial_latent: Tensor | None = None,
    dt_scale: float = 1.0,
    bos: bool | None = None,
    device: str | None = None,
) -> list[LoopStep]:
    """Run the Parcae recurrence for ``M`` loops and read out each loop.

    Parameters
    ----------
    model, tokenizer : from :func:`load_parcae`.
    prompt : the input text.
    M : number of recurrence loops to run (each yields one :class:`LoopStep`).
    initial_latent : optional starting recurrent state, shape
        ``(B, T, recurrent_embedding_dimension)``, matching the prompt's token count T.
        If ``None``, the model's own ``initialize_state`` is used (its default init).
    dt_scale : multiplier on the recurrence's learned input-injection step size ``dt``.
        ``1.0`` (default) is the in-distribution trained value. ``< 1`` slows convergence
        (the latent settles over more loops, useful for resolving basins); ``> 1`` speeds
        it up. Off-distribution when != 1 -- the fixed points shift. See :func:`_scaled_dt`.
        No-op for non-``diagonal`` injection types.
    bos : whether to prepend the tokenizer's BOS id, forwarded to ``tokenizer.encode``.
        ``None`` (default) uses the tokenizer's own default, i.e. **no** BOS -- correct for
        the released checkpoints, whose token 0 is untrained. Finetuned checkpoints trained
        with ``add_bos=True`` (e.g. the Countdown SFT runs) need ``bos=True``; omitting it
        there costs roughly 25x in accuracy.

    Returns
    -------
    list of :class:`LoopStep`, length ``M`` -- state + decoded answer after each loop.
    """
    device = device or next(model.parameters()).device.type
    was_training = model.training
    model.eval()

    input_ids = tokenizer.encode(prompt, device=device, bos=bos)
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)  # (1, T)

    input_embeds, freqs_cis = _run_prelude(model, input_ids)

    # --- initialize the recurrent state --------------------------------------
    if initial_latent is not None:
        x = initial_latent.to(device=input_embeds.device, dtype=input_embeds.dtype)
        expected = (
            input_embeds.shape[0],
            input_embeds.shape[1],
            model.config.recurrent_embedding_dimension,
        )
        if tuple(x.shape) != expected:
            raise ValueError(
                f"initial_latent shape {tuple(x.shape)} != expected {expected} "
                f"(batch, seq_len, recurrent_embedding_dimension)"
            )
    else:
        x = model.initialize_state(input_embeds)

    total_steps = torch.tensor(M, device=input_embeds.device)

    steps: list[LoopStep] = []
    with _scaled_dt(model, dt_scale):
        for k in range(M):
            xk = x
            step_t = torch.tensor(k, device=input_embeds.device)
            # one recurrence loop: the core block update (adapter injects input_embeds)
            x = model.update_recurrent_state(
                xk, input_embeds, freqs_cis, None, step_t, total_steps
            )

            residual = (x - xk).norm(dim=-1).mean().item()
            logits = _decode_latent(model, x, input_ids, freqs_cis)

            decoded_ids = logits[0].argmax(dim=-1).tolist()
            next_id = int(logits[0, -1].argmax().item())
            decoded_text = tokenizer.decode([next_id])

            steps.append(
                LoopStep(
                    loop=k + 1,
                    latent=x.detach().cpu(),
                    logits=logits.detach().cpu(),
                    decoded_ids=decoded_ids,
                    decoded_text=decoded_text,
                    residual=residual,
                )
            )

    if was_training:
        model.train()
    return steps


@dataclass
class BatchLoopStep:
    """State read out after one recurrence loop, across a batch of B conditions.

    Which tensors are populated depends on ``keep=`` in :func:`loop_trace_batched`;
    fields not requested are ``None`` to save memory. ``decoded_text`` is always kept.
    """

    loop: int  # 1-based loop index
    decoded_text: list[str]  # greedy next-token (last position) per condition, length B
    residual: Tensor | None = None  # per-condition ||z_k - z_{k-1}|| mean: (B,)
    decoded_ids: Tensor | None = None  # greedy argmax token id per position: (B, T)
    logits: Tensor | None = None  # next-token logits: (B, T, vocab), on CPU  (LARGE)
    latent: Tensor | None = None  # recurrent state: (B, T, d_rec), on CPU    (LARGE)
    extra: dict = field(default_factory=dict)


# what loop_trace_batched can retain per loop; the two "big" ones drive OOM.
_KEEP_FIELDS = ("decoded_text", "decoded_ids", "residual", "logits", "latent")


@torch.no_grad()
def loop_trace_batched(
    model,
    tokenizer,
    prompt: str,
    M: int,
    *,
    initial_latents: Tensor,
    keep=("decoded_text", "decoded_ids", "residual"),
    batch_size: int | None = None,
    dt_scale: float = 1.0,
    bos: bool | None = None,
    device: str | None = None,
) -> list[BatchLoopStep]:
    """Run the Parcae recurrence for ``M`` loops on B initial conditions at once.

    Same computation as calling :func:`loop_trace` B times, but the shared prelude is
    computed once and every recurrence/coda step runs on a ``(B, T, ...)`` batch, which
    is much faster than a Python loop over conditions (especially on GPU). Conditions
    never interact -- each row is an independent trajectory.

    Memory
    ------
    The usual OOM cause is *storing* per-loop tensors, not the model's working set.
    Kept ``logits`` cost ``B * T * vocab * 4 * M`` bytes and ``latent`` costs
    ``B * T * d_rec * 4 * M`` -- both scale with M and blow up for large B/T.

    * ``keep`` : which per-loop fields to retain. Defaults to the cheap ones
      (``decoded_text``, ``decoded_ids``, ``residual``); add ``"logits"`` / ``"latent"``
      only if you actually need them. ``decoded_text`` is always kept.
    * ``batch_size`` : if set, process the B conditions in chunks of this size and
      concatenate -- bounds peak *working* memory (the live logits/activation tensors)
      independent of total B. Results are identical to running all B at once.
    * ``dt_scale`` : multiplier on the recurrence's learned input-injection step size.
      ``1.0`` (default) is in-distribution; ``< 1`` slows convergence so the latent
      settles over more loops (useful for resolving basin structure), ``> 1`` speeds it
      up. Off-distribution when != 1. No-op for non-``diagonal`` injection. See
      :func:`_scaled_dt`.

    Parameters
    ----------
    initial_latents : starting recurrent states, shape
        ``(B, T, recurrent_embedding_dimension)`` where T is the prompt's token count.
        This is the batched analogue of ``loop_trace``'s ``initial_latent``.
    bos : whether to prepend the tokenizer's BOS id, forwarded to ``tokenizer.encode``.
        ``None`` (default) uses the tokenizer's own default, i.e. **no** BOS. Finetuned
        checkpoints trained with ``add_bos=True`` need ``bos=True``; see :func:`loop_trace`.
        Note this changes T, and hence the required ``initial_latents`` shape.

    Returns
    -------
    list of :class:`BatchLoopStep`, length ``M``. To match your current per-condition
    layout, stack ``decoded_text`` across loops and transpose:
        texts = np.array([s.decoded_text for s in trace]).T   # -> (B, M)
    """
    unknown = set(keep) - set(_KEEP_FIELDS)
    if unknown:
        raise ValueError(
            f"unknown keep field(s) {sorted(unknown)}; choose from {_KEEP_FIELDS}"
        )
    keep = set(keep) | {"decoded_text"}

    device = device or next(model.parameters()).device.type
    was_training = model.training
    model.eval()

    B = initial_latents.shape[0]
    expected_dim = model.config.recurrent_embedding_dimension

    ids = tokenizer.encode(prompt, device=device, bos=bos)
    if ids.dim() == 2:
        ids = ids[0]
    T = ids.shape[0]
    if tuple(initial_latents.shape) != (B, T, expected_dim):
        raise ValueError(
            f"initial_latents shape {tuple(initial_latents.shape)} != expected "
            f"{(B, T, expected_dim)} (batch, seq_len, recurrent_embedding_dimension)"
        )

    # --- optional chunking over the batch dim to bound peak working memory ----
    if batch_size is not None and batch_size < B:
        chunks = [
            loop_trace_batched(
                model,
                tokenizer,
                prompt,
                M,
                initial_latents=initial_latents[s : s + batch_size],
                keep=keep,
                batch_size=None,
                dt_scale=dt_scale,
                bos=bos,
                device=device,
            )
            for s in range(0, B, batch_size)
        ]
        merged: list[BatchLoopStep] = []
        for k in range(M):
            parts = [c[k] for c in chunks]
            merged.append(
                BatchLoopStep(
                    loop=k + 1,
                    decoded_text=[t for p in parts for t in p.decoded_text],
                    residual=torch.cat([p.residual for p in parts])
                    if "residual" in keep
                    else None,
                    decoded_ids=torch.cat([p.decoded_ids for p in parts])
                    if "decoded_ids" in keep
                    else None,
                    logits=torch.cat([p.logits for p in parts])
                    if "logits" in keep
                    else None,
                    latent=torch.cat([p.latent for p in parts])
                    if "latent" in keep
                    else None,
                )
            )
        if was_training:
            model.train()
        return merged

    input_ids = ids.unsqueeze(0).expand(B, -1).contiguous()  # (B, T)
    input_embeds, freqs_cis = _run_prelude(model, input_ids)
    x = initial_latents.to(device=input_embeds.device, dtype=input_embeds.dtype)

    total_steps = torch.tensor(M, device=input_embeds.device)

    steps: list[BatchLoopStep] = []
    with _scaled_dt(model, dt_scale):
        for k in range(M):
            xk = x
            step_t = torch.tensor(k, device=input_embeds.device)
            x = model.update_recurrent_state(
                xk, input_embeds, freqs_cis, None, step_t, total_steps
            )

            logits = _decode_latent(model, x, input_ids, freqs_cis)
            next_ids = logits[:, -1].argmax(dim=-1).tolist()  # last position per row
            decoded_text = [tokenizer.decode([int(t)]) for t in next_ids]

            steps.append(
                BatchLoopStep(
                    loop=k + 1,
                    decoded_text=decoded_text,
                    residual=(x - xk).norm(dim=-1).mean(dim=-1).cpu()
                    if "residual" in keep
                    else None,
                    decoded_ids=logits.argmax(dim=-1).cpu()
                    if "decoded_ids" in keep
                    else None,
                    logits=logits.detach().cpu() if "logits" in keep else None,
                    latent=x.detach().cpu() if "latent" in keep else None,
                )
            )

    if was_training:
        model.train()
    return steps


if __name__ == "__main__":
    # tiny smoke test / example
    mdl, tk = load_parcae("SandyResearch/parcae-140m")
    trace = loop_trace(mdl, tk, "The capital of France is", M=8)
    print(f"seq_len={trace[0].logits.shape[1]}, latent_dim={trace[0].latent.shape[-1]}")
    for s in trace:
        print(
            f"loop {s.loop:2d} | residual={s.residual:8.4f} | next-token={s.decoded_text!r}"
        )
