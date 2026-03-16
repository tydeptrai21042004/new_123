from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from math import gamma as math_gamma
from typing import Dict, List, Tuple

import numpy as np

from ..config import Prop2Config
from ..ops import SenseForwardOp, sampled_k_error, soft_data_consistency
from ..prox import temporal_fft_prox, wavelet_prox_video
from ..utils import mse
from .baselines import cg_sense_tikh

try:
    from scipy.special import gamma as sp_gamma
    from scipy.special import jv as sp_jv
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False


_EPS = 1e-12
_PI = np.pi


# -----------------------------------------------------------------------------
# Exact Hartley–Bessel ingredients
# -----------------------------------------------------------------------------

def _gamma(x: float) -> float:
    if _HAS_SCIPY:
        return float(sp_gamma(x))
    return float(math_gamma(x))


def _beta_bessel_series(z: np.ndarray, nu: float, terms: int = 40) -> np.ndarray:
    """
    beta_nu(z) = sum_{m>=0} (-1)^m / (m! (nu+1)_m) * (z/2)^(2m)
    Used as a fallback if scipy.special.jv is unavailable.
    """
    z = np.asarray(z, dtype=np.float64)
    out = np.ones_like(z, dtype=np.float64)
    term = np.ones_like(z, dtype=np.float64)
    zz = 0.25 * z * z

    for m in range(terms - 1):
        denom = float((m + 1) * (nu + 1.0 + m))
        term = term * (-zz / max(denom, _EPS))
        out = out + term

    return out


def _beta_bessel(z: np.ndarray, nu: float) -> np.ndarray:
    """
    Exact normalized Bessel combination:
        beta_nu(z) = Gamma(nu+1) * (2/z)^nu * J_nu(z),
    with the limit beta_nu(0)=1.

    This matches the series definition used in your manuscript.
    """
    z = np.asarray(z, dtype=np.float64)

    if not _HAS_SCIPY:
        return _beta_bessel_series(z, nu)

    out = np.empty_like(z, dtype=np.float64)
    small = np.abs(z) < 1e-12
    out[small] = 1.0

    if np.any(~small):
        zz = z[~small]
        out[~small] = _gamma(nu + 1.0) * np.power(2.0 / zz, nu) * sp_jv(nu, zz)

    return out


def _hartley_bessel_kernel(z: np.ndarray, alpha: float) -> np.ndarray:
    """
    J_lambda(t, alpha) = beta_{alpha-1/2}(lambda t)
                       + (lambda t)/(2 alpha + 1) * beta_{alpha+1/2}(lambda t)
    """
    z = np.asarray(z, dtype=np.float64)
    b1 = _beta_bessel(z, alpha - 0.5)
    b2 = _beta_bessel(z, alpha + 0.5)
    return b1 + (z / max(2.0 * alpha + 1.0, _EPS)) * b2


def _centered_indices(n: int) -> np.ndarray:
    """
    For n=8 -> [-4,-3,-2,-1,0,1,2,3]
    For n=7 -> [-3,-2,-1,0,1,2,3]
    """
    return np.arange(n, dtype=np.float64) - float(n // 2)


def _hb_measure_const(alpha: float) -> float:
    return 1.0 / (2.0 ** (alpha + 0.5) * _gamma(alpha + 0.5))


@dataclass(frozen=True)
class HB1DOp:
    n: int
    alpha: float
    h: float
    t: np.ndarray          # spatial nodes
    lam: np.ndarray        # frequency nodes
    mu_space: np.ndarray   # exact discrete spatial measure
    mu_freq: np.ndarray    # finite quadrature-like frequency measure
    kernel: np.ndarray     # K[k,n] = J_{lam_k}(t_n, alpha)
    analysis: np.ndarray   # A[k,n] = K[k,n] * mu_space[n]
    synthesis: np.ndarray  # S[n,k] = K[k,n] * mu_freq[k]
    inverse: np.ndarray    # numerically stable left inverse of analysis


@lru_cache(maxsize=32)
def _hb_1d_operator(n: int, alpha: float, h: float = 1.0) -> HB1DOp:
    """
    Build the exact finite Hartley–Bessel analysis operator on a sampled lattice.

    Spatial grid:
        t_n = n h, centered on zero

    Frequency grid:
        lambda_k = (2 pi / (N h)) k, centered on zero
        so the sampled band lies in [-pi/h, pi/h) up to finite-grid indexing.

    Analysis matrix:
        A[k,n] = J_{lambda_k}(t_n, alpha) * mu_alpha({t_n})

    Synthesis matrix:
        S[n,k] = J_{lambda_k}(t_n, alpha) * mu_alpha(d lambda_k)
    """
    alpha = float(alpha)
    h = float(h)

    idx_t = _centered_indices(n)
    t = idx_t * h

    delta_lam = 2.0 * _PI / (n * h)
    idx_lam = _centered_indices(n)
    lam = idx_lam * delta_lam

    c = _hb_measure_const(alpha)

    # exact discrete measure from the formula in your manuscript
    mu_space = h * c * np.power(np.abs(t), 2.0 * alpha)

    # finite-grid quadrature analogue of mu_alpha(d lambda)
    mu_freq = delta_lam * c * np.power(np.abs(lam), 2.0 * alpha)

    z = np.outer(lam, t)
    K = _hartley_bessel_kernel(z, alpha=alpha)

    A = K * mu_space[None, :]
    S = (K.T * mu_freq[None, :])

    # Numerically stable finite inverse for the truncated operator.
    # This does not change the exact analysis matrix; it only stabilizes
    # reconstruction on finite images.
    B = np.linalg.pinv(A, rcond=1e-8)

    return HB1DOp(
        n=n,
        alpha=alpha,
        h=h,
        t=t,
        lam=lam,
        mu_space=mu_space,
        mu_freq=mu_freq,
        kernel=K,
        analysis=A,
        synthesis=S,
        inverse=B,
    )


# -----------------------------------------------------------------------------
# Exact 2D Hartley–Bessel transform on images / video
# -----------------------------------------------------------------------------

def _hb2_forward_real(x: np.ndarray, op_y: HB1DOp, op_x: HB1DOp) -> np.ndarray:
    """
    Exact tensor-product 2D Hartley–Bessel analysis:
        Y = A_y X A_x^T
    x shape: [T, H, W]
    """
    return np.einsum("ph,thw,qw->tpq", op_y.analysis, x, op_x.analysis, optimize=True)


def _hb2_synthesis_real(y: np.ndarray, op_y: HB1DOp, op_x: HB1DOp) -> np.ndarray:
    """
    Direct synthesis induced by the inverse formula:
        X = S_y Y S_x^T
    """
    return np.einsum("hp,tpq,wq->thw", op_y.synthesis, y, op_x.synthesis, optimize=True)


def _hb2_inverse_real(y: np.ndarray, op_y: HB1DOp, op_x: HB1DOp) -> np.ndarray:
    """
    Stable finite inverse using pseudo-inverses of the exact analysis matrices:
        X = B_y Y B_x^T
    """
    return np.einsum("hp,tpq,wq->thw", op_y.inverse, y, op_x.inverse, optimize=True)


def _hb2_forward(x: np.ndarray, alpha: float) -> np.ndarray:
    """
    Exact 2D Hartley–Bessel transform applied separately to real/imag parts.
    """
    h, w = int(x.shape[-2]), int(x.shape[-1])
    op_y = _hb_1d_operator(h, alpha, 1.0)
    op_x = _hb_1d_operator(w, alpha, 1.0)

    xr = np.real(x)
    xi = np.imag(x)

    yr = _hb2_forward_real(xr, op_y, op_x)
    yi = _hb2_forward_real(xi, op_y, op_x)
    return yr + 1j * yi


def _hb2_inverse(y: np.ndarray, alpha: float) -> np.ndarray:
    """
    Stable inverse of the exact 2D Hartley–Bessel analysis.
    """
    h, w = int(y.shape[-2]), int(y.shape[-1])
    op_y = _hb_1d_operator(h, alpha, 1.0)
    op_x = _hb_1d_operator(w, alpha, 1.0)

    yr = np.real(y)
    yi = np.imag(y)

    xr = _hb2_inverse_real(yr, op_y, op_x)
    xi = _hb2_inverse_real(yi, op_y, op_x)
    return xr + 1j * xi


def _hb2_synthesis(y: np.ndarray, alpha: float) -> np.ndarray:
    """
    Direct synthesis corresponding to the quadrature analogue of the inverse formula.
    Useful for diagnostics / checking consistency.
    """
    h, w = int(y.shape[-2]), int(y.shape[-1])
    op_y = _hb_1d_operator(h, alpha, 1.0)
    op_x = _hb_1d_operator(w, alpha, 1.0)

    yr = np.real(y)
    yi = np.imag(y)

    xr = _hb2_synthesis_real(yr, op_y, op_x)
    xi = _hb2_synthesis_real(yi, op_y, op_x)
    return xr + 1j * xi


# -----------------------------------------------------------------------------
# Coefficient-domain mask / weighting
# -----------------------------------------------------------------------------

@lru_cache(maxsize=32)
def _hb_coeff_mask(
    h: int,
    w: int,
    alpha: float,
    protect_radius: float,
    outer_only: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build:
    - a radial mask in the exact HB coefficient domain
    - normalized coefficient quadrature weights

    The transform itself is exact; this mask is the regularization design
    that preserves your existing API semantics.
    """
    op_y = _hb_1d_operator(h, alpha, 1.0)
    op_x = _hb_1d_operator(w, alpha, 1.0)

    ly = op_y.lam
    lx = op_x.lam

    yy, xx = np.meshgrid(ly, lx, indexing="ij")
    r = np.sqrt(yy * yy + xx * xx)
    r = r / max(float(np.max(r)), _EPS)

    if outer_only:
        mask = np.clip((r - protect_radius) / max(1.0 - protect_radius, _EPS), 0.0, 1.0)
    else:
        mask = r.copy()
        if protect_radius > 0:
            inner = r <= protect_radius
            mask[inner] *= 0.15

    coeff_w = op_y.mu_freq[:, None] * op_x.mu_freq[None, :]
    coeff_w = coeff_w / max(float(np.mean(coeff_w)), _EPS)

    return np.asarray(mask, dtype=np.float64), np.asarray(coeff_w, dtype=np.float64)


def _soft_threshold_complex(a: np.ndarray, thresh: np.ndarray) -> np.ndarray:
    mag = np.abs(a)
    scale = np.maximum(0.0, 1.0 - thresh / np.maximum(mag, _EPS))
    return a * scale


def _hb_exact_shrink_video(
    x: np.ndarray,
    lam: float,
    alpha: float,
    protect_radius: float,
    outer_only: bool,
) -> np.ndarray:
    """
    Exact Hartley–Bessel analysis + coefficient shrinkage + stable inverse.

    This is the exact transform implementation requested.
    The only approximation left is the use of coefficient-domain shrinkage
    as a practical proximal surrogate, because the transform is not an
    exactly orthonormal FFT-like operator on finite truncated grids.
    """
    if lam <= 0.0:
        return x

    h, w = int(x.shape[-2]), int(x.shape[-1])
    mask, coeff_w = _hb_coeff_mask(h, w, alpha, protect_radius, outer_only)

    y = _hb2_forward(x, alpha=alpha)
    thresh = lam * mask[None, :, :] * coeff_w[None, :, :]
    y_sh = _soft_threshold_complex(y, thresh)
    x_out = _hb2_inverse(y_sh, alpha=alpha)
    return x_out


def _hb_exact_penalty(
    x: np.ndarray,
    alpha: float,
    protect_radius: float,
    outer_only: bool,
) -> float:
    """
    Penalty based on exact Hartley–Bessel coefficients.
    """
    h, w = int(x.shape[-2]), int(x.shape[-1])
    mask, coeff_w = _hb_coeff_mask(h, w, alpha, protect_radius, outer_only)
    y = _hb2_forward(x, alpha=alpha)
    val = np.mean(mask[None, :, :] * coeff_w[None, :, :] * np.abs(y))
    return float(val)


# -----------------------------------------------------------------------------
# Existing public API, now using the exact Hartley–Bessel transform
# -----------------------------------------------------------------------------

def objective_surrogate(
    op: SenseForwardOp,
    x: np.ndarray,
    y: np.ndarray,
    lam_w: float,
    lam_t: float,
    lam_h: float,
    cfg: Prop2Config,
) -> float:
    """
    API preserved.

    Note:
    - cfg.hartley_gamma is now interpreted as the exact Hartley–Bessel order alpha.
    """
    data = 0.5 * mse(op(x), y)

    dx1 = np.mean(np.abs(np.diff(x, axis=1))) if x.shape[1] > 1 else 0.0
    dx2 = np.mean(np.abs(np.diff(x, axis=2))) if x.shape[2] > 1 else 0.0
    wave_proxy = 0.5 * (dx1 + dx2)

    ft = np.fft.fft(np.moveaxis(x, 0, -1), axis=-1, norm="ortho")
    temp = (
        float(np.mean(np.abs(ft[..., cfg.temporal_keep_low_bins:])))
        if ft.shape[-1] > cfg.temporal_keep_low_bins
        else 0.0
    )

    hb_alpha = max(0.0, float(cfg.hartley_gamma))
    hart = _hb_exact_penalty(
        x,
        alpha=hb_alpha,
        protect_radius=float(cfg.hartley_protect_radius),
        outer_only=bool(cfg.outer_only_hartley),
    )

    return float(data + lam_w * wave_proxy + lam_t * temp + lam_h * hart)


def prop2_sense_pgd_improved(
    op: SenseForwardOp,
    y: np.ndarray,
    cfg: Prop2Config,
) -> Tuple[np.ndarray, List[Dict[str, float]]]:
    """
    API preserved.

    Main change:
    the previous Hartley radial prox is replaced by an exact Hartley–Bessel
    transform analysis / shrinkage / inverse block.
    """
    x = cg_sense_tikh(op, y, iters=10, lam=0.0015) if cfg.warm_start == "cg" else op.adjoint(y)
    z = x.copy()
    x_prev = x.copy()
    best_x = x.copy()
    best_score = float("inf")
    tau = cfg.tau0
    hist: List[Dict[str, float]] = []

    hb_alpha = max(0.0, float(cfg.hartley_gamma))

    for k in range(cfg.iters):
        frac = 0.0 if cfg.iters <= 1 else k / (cfg.iters - 1)
        alpha_mix = cfg.alpha0 + frac * (cfg.alpha1 - cfg.alpha0)
        dc_weight = cfg.dc_weight0 + frac * (cfg.dc_weight1 - cfg.dc_weight0)

        if cfg.continuation:
            lam_w = cfg.lam_w * (0.65 ** frac)
            lam_t = cfg.lam_t * (0.55 ** frac)
            lam_h = cfg.lam_h * (0.50 ** frac)
        else:
            lam_w, lam_t, lam_h = cfg.lam_w, cfg.lam_t, cfg.lam_h

        x_bar = z if cfg.use_nesterov else x
        grad = op.adjoint(op(x_bar) - y)
        base_obj = objective_surrogate(op, x_bar, y, lam_w, lam_t, lam_h, cfg)

        local_tau = tau
        x_candidate = x_bar

        for _ in range(8):
            x_half = x_bar - local_tau * grad

            x_reg = wavelet_prox_video(
                x_half,
                lam=lam_w * local_tau,
                wavelet=cfg.wavelet,
                level=cfg.wavelet_level,
            )

            x_reg = temporal_fft_prox(
                x_reg,
                lam=lam_t * local_tau,
                keep_low_bins=cfg.temporal_keep_low_bins,
                weighted=True,
            )

            # Exact Hartley–Bessel transform block
            x_reg = _hb_exact_shrink_video(
                x_reg,
                lam=lam_h * local_tau,
                alpha=hb_alpha,
                protect_radius=float(cfg.hartley_protect_radius),
                outer_only=bool(cfg.outer_only_hartley),
            )

            x_mix = (1.0 - alpha_mix) * x_half + alpha_mix * x_reg
            x_dc = soft_data_consistency(op, x_mix, y, d=dc_weight, exact=cfg.exact_dc)
            cand_obj = objective_surrogate(op, x_dc, y, lam_w, lam_t, lam_h, cfg)

            if (not cfg.monotone) or (cand_obj <= base_obj * 1.01):
                x_candidate = x_dc
                break

            local_tau = max(cfg.tau_min, local_tau * cfg.beta)

        tau = local_tau

        if cfg.use_restart and np.real(np.vdot((x_candidate - x), (x - x_prev))) > 0:
            z = x_candidate.copy()
        else:
            z = (
                x_candidate + ((k + 1) / (k + 4)) * (x_candidate - x)
                if cfg.use_nesterov
                else x_candidate.copy()
            )

        x_prev = x
        x = x_candidate

        err = sampled_k_error(op, x, y)
        obj = objective_surrogate(op, x, y, lam_w, lam_t, lam_h, cfg)

        hist.append(
            {
                "iter": int(k + 1),
                "tau": float(tau),
                "alpha": float(alpha_mix),
                "dc_weight": float(dc_weight),
                "lam_w": float(lam_w),
                "lam_t": float(lam_t),
                "lam_h": float(lam_h),
                "hb_alpha": float(hb_alpha),
                "sampled_k_err": float(err),
                "objective": float(obj),
            }
        )

        score = err + 0.02 * obj
        if score < best_score:
            best_score = score
            best_x = x.copy()

    return best_x, hist
