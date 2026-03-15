from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..ops import SenseForwardOp
from ..prox import low_rank_casorati_prox, temporal_fft_prox, wavelet_prox_video


@dataclass
class BaselineConfig:
    iters: int
    step: float
    lam: float




# SENSE + quadratic Tikhonov regularization.
def cg_sense_tikh(op: SenseForwardOp, y: np.ndarray, iters: int = 20, lam: float = 0.002) -> np.ndarray:
    b = op.adjoint(y)

    def H(z: np.ndarray) -> np.ndarray:
        return op.adjoint(op(z)) + lam * z

    x = np.zeros_like(b)
    r = b - H(x)
    p = r.copy()
    rsold = np.vdot(r, r)

    for _ in range(iters):
        Hp = H(p)
        denom = max(np.real(np.vdot(p, Hp)), 1e-12)
        alpha = rsold / denom
        x = x + alpha * p
        r = r - alpha * Hp
        rsnew = np.vdot(r, r)
        if np.sqrt(max(np.real(rsnew), 0.0)) < 1e-8:
            break
        p = r + (rsnew / max(rsold, 1e-12)) * p
        rsold = rsnew
    return x


# Sparse-MRI-style spatial wavelet sparsity baseline.
def fista_wavelet(op: SenseForwardOp, y: np.ndarray, cfg: BaselineConfig) -> np.ndarray:
    x = op.adjoint(y)
    z = x.copy()
    tk = 1.0
    for _ in range(cfg.iters):
        grad = op.adjoint(op(z) - y)
        x_next = wavelet_prox_video(z - cfg.step * grad, lam=cfg.lam * cfg.step)
        tk_next = 0.5 * (1.0 + math.sqrt(1.0 + 4.0 * tk * tk))
        z = x_next + ((tk - 1.0) / tk_next) * (x_next - x)
        x, tk = x_next, tk_next
    return x


# Dynamic-MRI temporal Fourier sparsity baseline.
def fista_tfft(op: SenseForwardOp, y: np.ndarray, cfg: BaselineConfig) -> np.ndarray:
    x = op.adjoint(y)
    z = x.copy()
    tk = 1.0
    for _ in range(cfg.iters):
        grad = op.adjoint(op(z) - y)
        x_next = temporal_fft_prox(z - cfg.step * grad, lam=cfg.lam * cfg.step, keep_low_bins=1, weighted=False)
        tk_next = 0.5 * (1.0 + math.sqrt(1.0 + 4.0 * tk * tk))
        z = x_next + ((tk - 1.0) / tk_next) * (x_next - x)
        x, tk = x_next, tk_next
    return x


# Low-rank dynamic MRI baseline via singular-value shrinkage on the Casorati matrix.
def pgd_lowrank(
    op: SenseForwardOp,
    y: np.ndarray,
    iters: int = 60,
    step: float = 0.15,
    lam0: float = 0.02,
    decay: float = 0.98,
) -> np.ndarray:
    x = op.adjoint(y)
    for k in range(iters):
        lam = lam0 * (decay ** k)
        grad = op.adjoint(op(x) - y)
        x = low_rank_casorati_prox(x - step * grad, lam=lam)
    return x
