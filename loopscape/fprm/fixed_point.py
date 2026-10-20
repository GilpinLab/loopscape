"""Fixed-point optimizer (FPOPT) — Algorithm 1 of the FPRM paper.

One damped fixed-point step with patience-based step-size decay.  State is a
dict carried across loop iterations:

    y        : current latent estimate  z_i            [B, S, D]
    residues : relative L_inf residual  r_i            [B]
    stepsize : damping factor eta                      [B]  (per-sample)
    best_res : smallest residual seen so far  r*        [B]
    patience : remaining patience counter   p          [B]

The map ``z_bar = f_theta(z; x)`` is computed *outside* this optimizer (by the
``FixedPointTransformer``); ``step`` only performs the damped update

    r     = ||y - z_bar||_inf / (||z_bar||_inf + eps)
    y_new = eta * z_bar + (1 - eta) * y

and decays ``eta <- gamma * eta`` after ``P`` consecutive iterations without
residual improvement (Algorithm 1, lines 3-12).
"""
import torch

from .config import FPTRMConfig


class FixedPointOptimizer:
    def __init__(self, config: FPTRMConfig):
        self.config = config

    # Read these live from config so they can be overridden per-call (e.g. via
    # solve_sudoku kwargs) after the model has been built.
    @property
    def eta0(self) -> float:
        return float(self.config.stepsize)

    @property
    def gamma(self) -> float:
        return float(self.config.stepsize_decay)

    @property
    def patience(self) -> int:
        return int(self.config.decay_patience)

    @property
    def eps(self) -> float:
        return float(self.config.eps)

    @property
    def tau(self) -> float:
        # FPOPT's step-size-decay guard uses its own nominal tau, decoupled from
        # the (possibly tighter) halting threshold ``fp_thresh``.
        return float(self.config.fpopt_tau)

    def reset(self, reset_flag, shape, dtype, device, state):
        """Initialise (or partially reset) the optimizer state.

        ``reset_flag`` is a per-sample bool tensor: where True the sample is
        (re)initialised, elsewhere the previous state is kept.
        """
        B = shape[0]
        zeros_y = torch.zeros(*shape, dtype=dtype, device=device)
        init_res = torch.full((B,), float("inf"), dtype=torch.float32, device=device)
        init_eta = torch.full((B,), self.eta0, dtype=torch.float32, device=device)
        init_pat = torch.full((B,), float(self.patience), dtype=torch.float32, device=device)

        if state is None:
            return {
                "y": zeros_y,
                "residues": init_res,
                "residues_state": init_res.clone(),
                "stepsize": init_eta,
                "best_res": init_res.clone(),
                "patience": init_pat,
            }

        flag_y = reset_flag.view((-1,) + (1,) * (zeros_y.ndim - 1))
        flag_v = reset_flag.view(-1)
        return {
            "y": torch.where(flag_y, zeros_y, state["y"]),
            "residues": torch.where(flag_v, init_res, state["residues"]),
            "residues_state": torch.where(flag_v, init_res, state["residues_state"]),
            "stepsize": torch.where(flag_v, init_eta, state["stepsize"]),
            "best_res": torch.where(flag_v, init_res, state["best_res"]),
            "patience": torch.where(flag_v, init_pat, state["patience"]),
        }

    def step(self, state, z_bar):
        """One damped fixed-point step.  ``z_bar = f_theta(y; x)``."""
        y = state["y"]
        eta = state["stepsize"]                       # [B]
        best = state["best_res"]                       # [B]
        pat = state["patience"]                        # [B]

        # Relative L_inf residual per sample.
        diff = (y - z_bar).abs().amax(dim=tuple(range(1, y.ndim)))      # [B]
        denom = z_bar.abs().amax(dim=tuple(range(1, y.ndim))) + self.eps
        r = (diff / denom).to(torch.float32)                            # [B]

        # Damped update.
        eta_b = eta.view((-1,) + (1,) * (y.ndim - 1)).to(y.dtype)
        y_new = eta_b * z_bar + (1.0 - eta_b) * y

        # Halting residual (Figure 3): relative change of the *damped* state,
        # ||z_i - z_{i-1}||_inf / ||z_i||_inf.  This is the smoothed signal the
        # fixed-point "Stop?" check uses.  It equals eta * r and so decreases as
        # the step size decays, giving a stable convergence indicator.
        state_diff = (y_new - y).abs().amax(dim=tuple(range(1, y.ndim)))
        state_denom = y_new.abs().amax(dim=tuple(range(1, y.ndim))) + self.eps
        r_state = (state_diff / state_denom).to(torch.float32)

        # Patience-based step-size decay (Algorithm 1, lines 5-11), tracked on
        # the undamped map residual r.
        improved = r < best
        best = torch.where(improved, r, best)
        pat = torch.where(improved, torch.full_like(pat, float(self.patience)), pat - 1.0)
        # Decay eta where patience exhausted and not yet converged.
        decay = (pat <= 0) & (r > self.tau)
        eta = torch.where(decay, eta * self.gamma, eta)
        pat = torch.where(decay, torch.full_like(pat, float(self.patience)), pat)

        return {
            "y": y_new,
            "residues": r,            # Algorithm 1 residual (used for halting + patience)
            "residues_state": r_state,  # damped-state change (Figure 3), kept for inspection
            "stepsize": eta,          # eta for the *next* step (post-decay)
            "dt": state["stepsize"],  # eta actually applied this step (the Euler dt)
            "best_res": best,
            "patience": pat,
        }

    @staticmethod
    def detach_state(state):
        return {k: (v.detach() if torch.is_tensor(v) else v) for k, v in state.items()}
