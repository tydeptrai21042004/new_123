from __future__ import annotations

from typing import Optional

import numpy as np
import pywt
from scipy.linalg import svd

from .ops import fft1c_time, fft2c, ifft1c_time, ifft2c
from .utils import complex_soft_threshold


def wavelet_prox_video(
    x: np.ndarray,
    lam: float,
    wavelet: str = "db4",
    level: int = 2,
    keep_approx: bool = True,
    scale_decay: float = 0.85,
) -> np.ndarray:
    out = np.empty_like(x)
    for tt in range(x.shape[0]):
        xr = np.real(x[tt])
        xi = np.imag(x[tt])

        coeffs_r = pywt.wavedec2(xr, wavelet=wavelet, level=level)
        coeffs_i = pywt.wavedec2(xi, wavelet=wavelet, level=level)

        def _shrink(coeffs):
            cA = coeffs[0]
            details = list(coeffs[1:])
            shrunk = []
            for s, (cH, cV, cD) in enumerate(details, start=1):
                fac = scale_decay ** (len(details) - s)
                thr = lam * fac
                shrunk.append(
                    (
                        np.sign(cH) * np.maximum(np.abs(cH) - thr, 0.0),
                        np.sign(cV) * np.maximum(np.abs(cV) - thr, 0.0),
                        np.sign(cD) * np.maximum(np.abs(cD) - 0.75 * thr, 0.0),
                    )
                )
            if keep_approx:
                cA2 = cA
            else:
                cA2 = np.sign(cA) * np.maximum(np.abs(cA) - 0.25 * lam, 0.0)
            return [cA2] + shrunk

        xr2 = pywt.waverec2(_shrink(coeffs_r), wavelet=wavelet)
        xi2 = pywt.waverec2(_shrink(coeffs_i), wavelet=wavelet)
        out[tt] = xr2[: x.shape[1], : x.shape[2]] + 1j * xi2[: x.shape[1], : x.shape[2]]
    return out


def temporal_fft_prox(x: np.ndarray, lam: float, keep_low_bins: int = 2, weighted: bool = True) -> np.ndarray:
    xt = np.moveaxis(x, 0, -1)
    ft = fft1c_time(xt)
    T = ft.shape[-1]
    freq = np.fft.fftfreq(T)
    weight = np.abs(freq)
    weight = 0.25 + 0.75 * weight / max(weight.max(), 1e-12) if weighted else np.ones_like(weight)

    if keep_low_bins > 0:
        protected = np.zeros(T, dtype=bool)
        protected[0] = True
        for k in range(1, min(keep_low_bins, T // 2 + 1)):
            protected[k] = True
            protected[-k] = True
        weight = weight.copy()
        weight[protected] = 0.0

    ft2 = complex_soft_threshold(ft, lam * weight[None, None, :])
    return np.moveaxis(ifft1c_time(ft2), -1, 0)


def radial_hartley_prox(
    x: np.ndarray,
    lam: float,
    outer_only: bool = True,
    gamma: float = 1.8,
    protect_radius: float = 0.20,
) -> np.ndarray:
    _, ny, nx = x.shape
    yy = np.linspace(-1.0, 1.0, ny)
    xx = np.linspace(-1.0, 1.0, nx)
    Y, X = np.meshgrid(yy, xx, indexing="ij")
    r = np.sqrt(X ** 2 + Y ** 2)

    w = np.clip((r - protect_radius) / max(1e-12, 1.0 - protect_radius), 0.0, 1.0) ** gamma
    if not outer_only:
        w = 0.10 + 0.90 * r ** gamma

    F = fft2c(x)
    H = np.real(F) - np.imag(F)
    H2 = np.sign(H) * np.maximum(np.abs(H) - lam * w[None, :, :], 0.0)

    ratio = np.ones_like(H, dtype=np.float32)
    nz = np.abs(H) > 1e-9
    ratio[nz] = H2[nz] / H[nz]
    return ifft2c(ratio * F)


def low_rank_casorati_prox(x: np.ndarray, lam: float, rank_cap: Optional[int] = None) -> np.ndarray:
    t, ny, nx = x.shape
    X = x.reshape(t, ny * nx).T
    U, s, Vh = svd(X, full_matrices=False)
    s2 = np.maximum(s - lam, 0.0)
    if rank_cap is not None:
        s2[rank_cap:] = 0.0
    return ((U * s2[None, :]) @ Vh).T.reshape(t, ny, nx)
