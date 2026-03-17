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


def _beta_bessel_series(z: np.ndarray, nu: float, terms: int = 60) -> np.ndarray:
    """
    beta_nu(z) = sum_{m>=0} (-1)^m / (m! (nu+1)_m) * (z/2)^(2m)

    Stable fallback / small-argument evaluator.
    """
    z = np.asarray(z, dtype=np.float64)
    out = np.ones_like(z, dtype=np.float64)
    term = np.ones_like(z, dtype=np.float64)
    zz = 0.25 * z * z

    for m in range(terms - 1):
        denom = float((m + 1) * (nu + 1.0 + m))
        denom = max(denom, _EPS)
        term = term * (-zz / denom)
        out = out + term

    return out


def _beta_bessel(z: np.ndarray, nu: float) -> np.ndarray:
    """
    Stable real-valued evaluation of

        beta_nu(z) = Gamma(nu+1) * (2/|z|)^nu * J_nu(|z|),

    with beta_nu(0)=1.

    Important:
    - use abs(z), because the series definition is even in z,
      while (2/z)^nu is not real-valued for z<0 and noninteger nu.
    """
    z = np.asarray(z, dtype=np.float64)
    az = np.abs(z)

    out = np.ones_like(az, dtype=np.float64)
    small = az < 1e-6

    if np.any(small):
        out[small] = _beta_bessel_series(az[small], nu, terms=60)

    if np.any(~small):
        if _HAS_SCIPY:
            zz = az[~small]
            tmp = _gamma(nu + 1.0) * np.power(2.0 / zz, nu) * sp_jv(nu, zz)
            bad = ~np.isfinite(tmp)
            if np.any(bad):
                tmp[bad] = _beta_bessel_series(zz[bad], nu, terms=60)
            out[~small] = tmp
        else:
            out[~small] = _beta_bessel_series(az[~small], nu, terms=60)

    return np.nan_to_num(out, nan=1.0, posinf=0.0, neginf=0.0)


def _hartley_bessel_kernel(z: np.ndarray, alpha: float) -> np.ndarray:
    """
    J_lambda(t, alpha) = beta_{alpha-1/2}(lambda t)
                       + (lambda t)/(2 alpha + 1) * beta_{alpha+1/2}(lambda t)
    """
    z = np.asarray(z, dtype=np.float64)
    b1 = _beta_bessel(z, alpha - 0.5)
    b2 = _beta_bessel(z, alpha + 0.5)
    out = b1 + (z / max(2.0 * alpha + 1.0, _EPS)) * b2
    return np.nan_to_num(out, nan=1.0, posinf=0.0, neginf=0.0)


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
    t: np.ndarray
    lam: np.ndarray
    mu_space: np.ndarray
    mu_freq: np.ndarray
    kernel: np.ndarray
    analysis: np.ndarray
    synthesis: np.ndarray
    inverse: np.ndarray


def _regularized_left_inverse(A: np.ndarray, ridge_scale: float = 1e-6) -> np.ndarray:
    """
    Compute a stable left inverse:
        B = (A^T A + ridge I)^(-1) A^T
    """
    n = A.shape[1]
    scale = float(np.linalg.norm(A, ord="fro"))
    ridge = max(1e-8, ridge_scale * (scale * scale / max(n, 1) + 1.0))

    G = A.T @ A
    G = 0.5 * (G + G.T)
    G.flat[:: n + 1] += ridge

    try:
        B = np.linalg.solve(G, A.T)
    except np.linalg.LinAlgError:
        B = np.linalg.lstsq(G, A.T, rcond=None)[0]

    return np.nan_to_num(B, nan=0.0, posinf=0.0, neginf=0.0)


@lru_cache(maxsize=32)
def _hb_1d_operator(n: int, alpha: float, h: float = 1.0) -> HB1DOp:
    """
    Build a finite Hartley–Bessel analysis operator on a centered sampled lattice.

    Analysis matrix:
        A[k,n] = J_{lambda_k}(t_n, alpha) * mu_alpha({t_n})

    Synthesis matrix:
        S[n,k] = J_{lambda_k}(t_n, alpha) * mu_alpha(d lambda_k)

    The stored inverse is a stable regularized left inverse.
    """
    alpha = float(alpha)
    h = float(h)

    idx_t = _centered_indices(n)
    t = idx_t * h

    delta_lam = 2.0 * _PI / (n * h)
    idx_lam = _centered_indices(n)
    lam = idx_lam * delta_lam

    c = _hb_measure_const(alpha)

    abs_t = np.abs(t)
    abs_lam = np.abs(lam)

    mu_space = h * c * np.power(abs_t, 2.0 * alpha)
    mu_freq = delta_lam * c * np.power(abs_lam, 2.0 * alpha)

    # Clean origin when alpha = 0
    if np.isclose(alpha, 0.0):
        mu_space[np.isclose(abs_t, 0.0)] = h * c
        mu_freq[np.isclose(abs_lam, 0.0)] = delta_lam * c

    z = np.outer(lam, t)
    K = _hartley_bessel_kernel(z, alpha=alpha)
    K = np.nan_to_num(K, nan=1.0, posinf=0.0, neginf=0.0)

    A = K * mu_space[None, :]
    S = K.T * mu_freq[None, :]

    A = np.nan_to_num(A, nan=0.0, posinf=0.0, neginf=0.0)
    S = np.nan_to_num(S, nan=0.0, posinf=0.0, neginf=0.0)

    B = _regularized_left_inverse(A)

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
    Stable finite inverse using regularized left inverses:
        X = B_y Y B_x^T
    """
    return np.einsum("hp,tpq,wq->thw", op_y.inverse, y, op_x.inverse, optimize=True)


def _hb2_forward(x: np.ndarray, alpha: float) -> np.ndarray:
    """
    2D Hartley–Bessel transform applied separately to real/imag parts.
    """
    h, w = int(x.shape[-2]), int(x.shape[-1])
    op_y = _hb_1d_operator(h, alpha, 1.0)
    op_x = _hb_1d_operator(w, alpha, 1.0)

    xr = np.real(x)
    xi = np.imag(x)

    yr = _hb2_forward_real(xr, op_y, op_x)
    yi = _hb2_forward_real(xi, op_y, op_x)

    out = yr + 1j * yi
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def _hb2_inverse(y: np.ndarray, alpha: float) -> np.ndarray:
    """
    Stable inverse of the finite 2D Hartley–Bessel analysis.
    """
    h, w = int(y.shape[-2]), int(y.shape[-1])
    op_y = _hb_1d_operator(h, alpha, 1.0)
    op_x = _hb_1d_operator(w, alpha, 1.0)

    yr = np.real(y)
    yi = np.imag(y)

    xr = _hb2_inverse_real(yr, op_y, op_x)
    xi = _hb2_inverse_real(yi, op_y, op_x)

    out = xr + 1j * xi
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


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

    out = xr + 1j * xi
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


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
    - a radial mask in the HB coefficient domain
    - normalized coefficient quadrature weights
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

    mask = np.nan_to_num(mask, nan=0.0, posinf=1.0, neginf=0.0)
    coeff_w = np.nan_to_num(coeff_w, nan=0.0, posinf=0.0, neginf=0.0)

    return np.asarray(mask, dtype=np.float64), np.asarray(coeff_w, dtype=np.float64)


def _soft_threshold_complex(a: np.ndarray, thresh: np.ndarray) -> np.ndarray:
    mag = np.abs(a)
    scale = np.maximum(0.0, 1.0 - thresh / np.maximum(mag, _EPS))
    out = a * scale
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def _hb_exact_shrink_video(
    x: np.ndarray,
    lam: float,
    alpha: float,
    protect_radius: float,
    outer_only: bool,
) -> np.ndarray:
    """
    Exact Hartley–Bessel analysis + coefficient shrinkage + stable inverse.
    """
    if lam <= 0.0:
        return x

    h, w = int(x.shape[-2]), int(x.shape[-1])
    mask, coeff_w = _hb_coeff_mask(h, w, alpha, protect_radius, outer_only)

    y = _hb2_forward(x, alpha=alpha)
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)

    thresh = lam * mask[None, :, :] * coeff_w[None, :, :]
    y_sh = _soft_threshold_complex(y, thresh)

    x_out = _hb2_inverse(y_sh, alpha=alpha)
    return np.nan_to_num(x_out, nan=0.0, posinf=0.0, neginf=0.0)


def _hb_exact_penalty(
    x: np.ndarray,
    alpha: float,
    protect_radius: float,
    outer_only: bool,
) -> float:
    """
    Penalty based on Hartley–Bessel coefficients.
    """
    h, w = int(x.shape[-2]), int(x.shape[-1])
    mask, coeff_w = _hb_coeff_mask(h, w, alpha, protect_radius, outer_only)

    y = _hb2_forward(x, alpha=alpha)
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)

    val = np.mean(mask[None, :, :] * coeff_w[None, :, :] * np.abs(y))
    return float(np.nan_to_num(val, nan=0.0, posinf=0.0, neginf=0.0))


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
    - cfg.hartley_gamma is interpreted as the Hartley–Bessel order alpha.
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
    the Hartley block uses exact Hartley–Bessel analysis / shrinkage / inverse.
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
