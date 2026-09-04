"""Model configuration for FPRM (single-Z fixed-point reasoning model).

Fields and defaults are taken from ``fprm_weights/sudoku/all_config.yaml`` (the
``arch`` block) and the usage in the release's ``fp_trm_singlez.py``.  Only fields touched
on the inference path are required; the rest are kept for faithfulness and to
allow constructing the config directly from the yaml.
"""
from dataclasses import dataclass
from typing import Optional


@dataclass
class FPTRMConfig:
    # Sizes
    vocab_size: int = 11
    seq_len: int = 81
    hidden_size: int = 512
    num_heads: int = 8
    expansion: float = 4.0

    # Reasoning layers
    L_layers: int = 2
    L_cycles: int = 6           # informational; loop count is governed by the FP optimizer at eval
    H_layers: int = 0
    H_cycles: int = 3

    # Puzzle embedding
    puzzle_emb_ndim: int = 512
    puzzle_emb_len: int = 16
    num_puzzle_identifiers: int = 1
    batch_size: int = 768       # only used to size training-only buffers (unused at inference)

    # Positional encoding
    pos_encodings: str = "rope"
    rope_theta: float = 10000.0

    # Residual scaling / fixed-point dynamics
    alpha_1_init: float = 0.75
    alpha_2_init: float = 0.25
    scale_init: float = 2.0
    residual_scale: str = "input-independent"
    normalize_input_injection: bool = False

    # Depthwise short-conv
    conv_type: str = "conv2d"     # "conv2d" for Sudoku/ARC (2D grid), "conv1d" otherwise
    conv_kernel_size: int = 3
    conv_bias: bool = False

    # Norm
    norm_type: str = "pre-norm"
    norm_placement: str = "none"
    rms_norm_eps: float = 1e-8

    # Decoding
    n_decode_steps: int = 0

    # Halting / fixed-point optimizer (FPOPT)
    halting_mechanism: str = "fixed_point"
    fp_thresh: float = 0.1          # residual tolerance tau for the *halting* check
    # tau used by FPOPT's step-size decay guard (Algorithm 1, "r > tau").  Kept at
    # the paper's nominal 0.1 and decoupled from fp_thresh so that tightening the
    # halting tolerance does not change the eta-decay schedule (and hence the
    # latent trajectory).
    fpopt_tau: float = 0.1
    stepsize: float = 1.0           # initial damping eta_0
    stepsize_decay: float = 0.9     # geometric decay gamma
    decay_patience: int = 5         # patience P
    eps: float = 1e-8
    max_iter: int = 12
    max_iter_eval: Optional[int] = 1000
    max_iter_dist: str = "det"
    n_backwards_L: int = 6          # training only

    # Dtype
    forward_dtype: str = "bfloat16"

    # Training-only knobs (kept so the yaml can be splatted in)
    variational_dropout: float = 0.0
    weight_dropout: float = 0.0
    no_ACT_continue: bool = True
    halt_max_steps: int = 16
    halt_exploration_prob: float = 0.0
    softmax_temp: float = 1.0
    mlp_t: bool = False
    outer_only: bool = False
    outlier_quantile: float = 0.25
    gamma_alpha: float = 4.0
    gamma_scale: float = 16.6666666667
    expon_scale: float = 72.1347520444
    scale_decay: float = 0.9

    def __init__(self, **kwargs):
        # Accept the full yaml ``arch`` dict (with extra keys) without erroring.
        import dataclasses
        known = {f.name for f in dataclasses.fields(self)}
        for name in known:
            setattr(self, name, kwargs.get(name, getattr(type(self), name, None)))
