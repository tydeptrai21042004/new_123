from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from ..config import Prop2Config
from ..ops import SenseForwardOp, sampled_k_error, soft_data_consistency
from ..prox import radial_hartley_prox, temporal_fft_prox, wavelet_prox_video
from ..utils import mse
from .baselines import cg_sense_tikh


def objective_surrogate(
    op: SenseForwardOp,
    x: np.ndarray,
    y: np.ndarray,
    lam_w: float,
    lam_t: float,
    lam_h: float,
    cfg: Prop2Config,
) -> float:
    data = 0.5 * mse(op(x), y)
    dx1 = np.mean(np.abs(np.diff(x, axis=1))) if x.shape[1] > 1 else 0.0
    dx2 = np.mean(np.abs(np.diff(x, axis=2))) if x.shape[2] > 1 else 0.0
    wave_proxy = 0.5 * (dx1 + dx2)
    ft = np.fft.fft(np.moveaxis(x, 0, -1), axis=-1, norm="ortho")
    temp = float(np.mean(np.abs(ft[..., cfg.temporal_keep_low_bins :]))) if ft.shape[-1] > cfg.temporal_keep_low_bins else 0.0
    F = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(x, axes=(-2, -1)), norm="ortho"), axes=(-2, -1))
    H = np.real(F) - np.imag(F)
    hart = float(np.mean(np.abs(H)))
    return float(data + lam_w * wave_proxy + lam_t * temp + lam_h * hart)


def prop2_sense_pgd_improved(op: SenseForwardOp, y: np.ndarray, cfg: Prop2Config) -> Tuple[np.ndarray, List[Dict[str, float]]]:
    x = cg_sense_tikh(op, y, iters=10, lam=0.0015) if cfg.warm_start == "cg" else op.adjoint(y)
    z = x.copy()
    x_prev = x.copy()
    best_x = x.copy()
    best_score = float("inf")
    tau = cfg.tau0
    hist: List[Dict[str, float]] = []

    for k in range(cfg.iters):
        frac = 0.0 if cfg.iters <= 1 else k / (cfg.iters - 1)
        alpha = cfg.alpha0 + frac * (cfg.alpha1 - cfg.alpha0)
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
            x_reg = radial_hartley_prox(
                x_reg,
                lam=lam_h * local_tau,
                outer_only=cfg.outer_only_hartley,
                gamma=cfg.hartley_gamma,
                protect_radius=cfg.hartley_protect_radius,
            )

            x_mix = (1.0 - alpha) * x_half + alpha * x_reg
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
            z = x_candidate + ((k + 1) / (k + 4)) * (x_candidate - x) if cfg.use_nesterov else x_candidate.copy()

        x_prev = x
        x = x_candidate

        err = sampled_k_error(op, x, y)
        obj = objective_surrogate(op, x, y, lam_w, lam_t, lam_h, cfg)
        hist.append(
            {
                "iter": int(k + 1),
                "tau": float(tau),
                "alpha": float(alpha),
                "dc_weight": float(dc_weight),
                "lam_w": float(lam_w),
                "lam_t": float(lam_t),
                "lam_h": float(lam_h),
                "sampled_k_err": float(err),
                "objective": float(obj),
            }
        )

        score = err + 0.02 * obj
        if score < best_score:
            best_score = score
            best_x = x.copy()

    return best_x, hist
