from __future__ import annotations

import math
from typing import Dict, Optional

import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def rss(x: np.ndarray, axis: int = 0, eps: float = 1e-12) -> np.ndarray:
    return np.sqrt(np.sum(np.abs(x) ** 2, axis=axis) + eps)


def mse(a: np.ndarray, b: np.ndarray) -> float:
    d = np.asarray(a) - np.asarray(b)
    return float(np.mean(np.abs(d) ** 2))


def nrmse_euclidean(ref: np.ndarray, test: np.ndarray, eps: float = 1e-12) -> float:
    ref = np.asarray(ref)
    test = np.asarray(test)
    return float(np.linalg.norm(ref - test) / (np.linalg.norm(ref) + eps))


def center_crop(x: np.ndarray, crop_fraction: float = 0.5) -> np.ndarray:
    ny, nx = x.shape[-2:]
    chy = max(1, int(round(ny * crop_fraction)))
    chx = max(1, int(round(nx * crop_fraction)))
    y0 = (ny - chy) // 2
    x0 = (nx - chx) // 2
    return x[..., y0 : y0 + chy, x0 : x0 + chx]


def _to_real_image(x: np.ndarray) -> np.ndarray:
    return np.abs(x) if np.iscomplexobj(x) else np.asarray(x)


def compute_metrics(ref: np.ndarray, pred: np.ndarray, data_range: Optional[float] = None) -> Dict[str, float]:
    ref_r = _to_real_image(ref)
    pred_r = _to_real_image(pred)
    if data_range is None:
        data_range = float(ref_r.max() - ref_r.min() + 1e-12)
    return {
        "PSNR": float(peak_signal_noise_ratio(ref_r, pred_r, data_range=data_range)),
        "SSIM": float(
            structural_similarity(
                ref_r,
                pred_r,
                data_range=data_range,
                gaussian_weights=True,
                sigma=1.5,
                use_sample_covariance=False,
            )
        ),
        "NRMSE": nrmse_euclidean(ref_r, pred_r),
    }


def complex_soft_threshold(z: np.ndarray, lam: np.ndarray | float) -> np.ndarray:
    mag = np.abs(z)
    lam = np.asarray(lam)
    scale = np.maximum(0.0, 1.0 - lam / np.maximum(mag, 1e-12))
    return scale * z


def format_pm(mean: float, std: float, digits: int = 4) -> str:
    return f"{mean:.{digits}f} ± {std:.{digits}f}"


def add_complex_noise(k: np.ndarray, mask: np.ndarray, rel_std: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = (rng.standard_normal(k.shape) + 1j * rng.standard_normal(k.shape)) / math.sqrt(2.0)
    obs = k[..., mask > 0]
    sigma_base = float(np.std(obs)) if obs.size > 0 else 1.0
    sigma = rel_std * max(sigma_base, 1e-12)
    return k + sigma * n.astype(np.complex64)
