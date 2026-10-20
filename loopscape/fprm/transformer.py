"""The looped fixed-point map f_theta(z; x) — ``L_level`` in the checkpoint.

Implements the FPRM layer of Figure 3 / Eqs (2)-(3) and Appendix A.1:

  * A depthwise short-convolution applied to the latent at the start of each
    loop (Appendix C, "depth-wise convolutions ... at the beginning of each
    loop"; 2D variant for the 9x9 Sudoku grid).
  * ``L_layers`` pre-norm Transformer layers.  Each layer has two sub-layers
    (attention, then SwiGLU MLP); for sub-layer ell the residual stream and
    sub-layer output are combined with tied per-channel scalars (alpha_1, beta_1):

        z^ell = alpha_1 * z^(ell-1) + beta_1 * f_ell(Norm_pre(z^(ell-1)))      (Eq 2)

  * Iteration-wise input mixing with tied scalars (alpha_2, beta_2):

        z_bar = alpha_2 * z^2L + beta_2 * x                                     (Eq 3)

beta_1 and beta_2 are *derived* from alpha_1, alpha_2 (Theorem 1 / Appendix A.1):

        beta_2 = 1 - alpha_2 * alpha_1^(2L)
        beta_1 = beta_2 * (1 - alpha_1) / (1 - alpha_1^(2L))

alpha_1, alpha_2 are obtained from the learnable per-channel parameters via a
sigmoid so they stay in (0, 1) as required by the boundedness theorem.
"""
import torch
import torch.nn.functional as F
from torch import nn

from .config import FPTRMConfig
from .layers import Attention, SwiGLU, rms_norm


class TransformerBlock(nn.Module):
    """One pre-norm Transformer layer = attention sub-layer + MLP sub-layer."""

    def __init__(self, config: FPTRMConfig):
        super().__init__()
        head_dim = config.hidden_size // config.num_heads
        self.self_attn = Attention(
            hidden_size=config.hidden_size,
            head_dim=head_dim,
            num_heads=config.num_heads,
            num_key_value_heads=config.num_heads,
            causal=False,
        )
        self.mlp = SwiGLU(hidden_size=config.hidden_size, expansion=config.expansion)
        self.eps = config.rms_norm_eps


class FixedPointTransformer(nn.Module):
    def __init__(self, config: FPTRMConfig, n_layers: int):
        super().__init__()
        self.config = config
        self.n_layers = n_layers
        # 2L sub-layers (attention + MLP per layer).
        self.num_sublayers = 2 * n_layers
        self.eps = config.rms_norm_eps

        self.layers = nn.ModuleList([TransformerBlock(config) for _ in range(n_layers)])

        # Per-channel residual-scaling parameters (sigmoid -> alpha in (0, 1)).
        self.alpha_1_param = nn.Parameter(torch.zeros(config.hidden_size))
        self.alpha_2_param = nn.Parameter(torch.zeros(config.hidden_size))

        # Depthwise short-conv over the puzzle grid.
        k = config.conv_kernel_size
        if config.conv_type == "conv2d":
            self.conv = nn.Conv2d(
                config.hidden_size, config.hidden_size, kernel_size=k,
                padding=k // 2, groups=config.hidden_size, bias=config.conv_bias,
            )
        else:  # conv1d
            self.conv = nn.Conv1d(
                config.hidden_size, config.hidden_size, kernel_size=k,
                padding=k // 2, groups=config.hidden_size, bias=config.conv_bias,
            )

    # ------------------------------------------------------------------ alphas
    def _alphas(self):
        a1 = torch.sigmoid(self.alpha_1_param.float())
        a2 = torch.sigmoid(self.alpha_2_param.float())
        a1_2L = a1 ** self.num_sublayers
        beta_2 = 1.0 - a2 * a1_2L
        # Guard the (1 - a1) / (1 - a1^2L) ratio against a1 -> 1.
        denom = (1.0 - a1_2L).clamp_min(1e-12)
        beta_1 = beta_2 * (1.0 - a1) / denom
        return a1, a2, beta_1, beta_2

    # ----------------------------------------------------------------- shortconv
    def _short_conv(self, z, puzzle_emb_len):
        """Depthwise conv over the grid tokens, leaving prefix tokens untouched."""
        prefix = z[:, :puzzle_emb_len]
        grid = z[:, puzzle_emb_len:]                       # [B, N, D]
        B, N, D = grid.shape
        if self.config.conv_type == "conv2d":
            side = int(round(N ** 0.5))                    # 9 for Sudoku
            x = grid.transpose(1, 2).reshape(B, D, side, side)
            x = self.conv(x.to(self.conv.weight.dtype))
            grid = x.reshape(B, D, N).transpose(1, 2)
        else:
            # Causal 1D short-conv (Maze/state-tracking): left-pad by k-1 and use
            # padding=0 so the output length is exactly N for *any* kernel size.
            # The conv module was built with symmetric padding=k//2, which for an
            # even kernel (Maze uses k=4) would emit N+1 tokens; do the padding
            # ourselves via functional conv instead (weights are identical).
            k = self.config.conv_kernel_size
            x = grid.transpose(1, 2).to(self.conv.weight.dtype)   # [B, D, N]
            x = F.pad(x, (k - 1, 0))
            x = F.conv1d(x, self.conv.weight, self.conv.bias,
                         padding=0, groups=self.config.hidden_size)
            grid = x.transpose(1, 2)
        return torch.cat((prefix, grid.to(z.dtype)), dim=1)

    # -------------------------------------------------------------------- forward
    def forward(self, z, input_injection, cos_sin=None, puzzle_emb_len=0):
        """Compute one fixed-point map z_bar = f_theta(z; x).

        Following Figure 3, the iteration-wise input mixing happens *before* the
        Transformer layers:  the latent is short-convolved, scaled by alpha_2 and
        combined with the (beta_2-scaled) input injection x, and the result is fed
        through the pre-norm layers (which apply the layer-wise alpha_1/beta_1
        residual scaling internally).
        """
        a1, a2, beta_1, beta_2 = self._alphas()

        # Short-conv on the latent, then iteration-wise input mixing (Eq 3).
        h = self._short_conv(z, puzzle_emb_len)
        h = a2 * h + beta_2 * input_injection

        # Pre-norm layers with layer-wise residual scaling (Eq 2), applied
        # per sub-layer (attention, then MLP, for each layer).
        for layer in self.layers:
            attn_out = layer.self_attn(cos_sin, rms_norm(h, self.eps))
            h = a1 * h + beta_1 * attn_out
            mlp_out = layer.mlp(rms_norm(h, self.eps))
            h = a1 * h + beta_1 * mlp_out

        # norm_placement="output" (Maze recipe): RMS-norm the loop's output before
        # it is returned to the fixed-point optimizer.  Sudoku uses "none" (no-op).
        if getattr(self.config, "norm_placement", "none") == "output":
            h = rms_norm(h, self.eps)

        return h
