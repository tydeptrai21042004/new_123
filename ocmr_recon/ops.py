from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter

from .utils import center_crop, rss


def fft2c(x: np.ndarray) -> np.ndarray:
    return np.fft.fftshift(
        np.fft.fft2(np.fft.ifftshift(x, axes=(-2, -1)), norm="ortho"),
        axes=(-2, -1),
    )


def ifft2c(x: np.ndarray) -> np.ndarray:
    return np.fft.fftshift(
        np.fft.ifft2(np.fft.ifftshift(x, axes=(-2, -1)), norm="ortho"),
        axes=(-2, -1),
    )


def fft1c_time(x: np.ndarray) -> np.ndarray:
    return np.fft.fft(x, axis=-1, norm="ortho")


def ifft1c_time(x: np.ndarray) -> np.ndarray:
    return np.fft.ifft(x, axis=-1, norm="ortho")


def make_reference_from_full_kspace(full_kspace: np.ndarray, crop_fraction: float = 0.5) -> np.ndarray:
    return center_crop(rss(ifft2c(full_kspace), axis=0), crop_fraction)


def make_reference_sense(full_kspace: np.ndarray, sens: np.ndarray, crop_fraction: float = 0.5) -> np.ndarray:
    coil_imgs = ifft2c(full_kspace)
    x_ref = np.sum(np.conj(sens)[:, None, :, :] * coil_imgs, axis=0)
    return center_crop(np.abs(x_ref), crop_fraction)


def estimate_sens_maps_from_acs(
    full_kspace: np.ndarray,
    acs_lines: int = 12,
    ky_center: Optional[int] = None,
    smooth_eps: float = 1e-6,
) -> np.ndarray:
    _, _, ny, _ = full_kspace.shape
    cy = int(np.clip(ky_center if ky_center is not None else ny // 2, 0, ny - 1))
    lo = max(0, cy - acs_lines // 2)
    hi = min(ny, cy + int(np.ceil(acs_lines / 2.0)))

    k = np.zeros_like(full_kspace)
    k[:, :, lo:hi, :] = full_kspace[:, :, lo:hi, :]

    imgs = ifft2c(k)
    mean_img = np.mean(imgs, axis=1)

    mean_img_s = (
        gaussian_filter(np.real(mean_img), sigma=(0, 1.2, 1.2))
        + 1j * gaussian_filter(np.imag(mean_img), sigma=(0, 1.2, 1.2))
    )

    denom = np.sqrt(np.sum(np.abs(mean_img_s) ** 2, axis=0))
    denom = np.maximum(denom, smooth_eps)

    sens = mean_img_s / denom[None, ...]
    return sens.astype(np.complex64)


def build_vd_mask(
    ny: int,
    nx: int,
    R: float,
    acs_lines: int = 12,
    seed: int = 0,
    ky_center: Optional[int] = None,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    cy = int(np.clip(ky_center if ky_center is not None else ny // 2, 0, ny - 1))
    mask_1d = np.zeros(ny, dtype=np.float32)

    acs_lo = max(0, cy - acs_lines // 2)
    acs_hi = min(ny, cy + int(np.ceil(acs_lines / 2.0)))
    mask_1d[acs_lo:acs_hi] = 1.0

    target_lines = max(acs_lines, int(round(ny / R)))
    remain = max(0, target_lines - int(mask_1d.sum()))

    yy = np.arange(ny)
    dist = np.abs((yy - cy) / max(max(cy, ny - 1 - cy), 1))
    pdf = np.exp(-4.0 * dist ** 2)
    pdf[acs_lo:acs_hi] = 0.0

    pdf_sum = float(pdf.sum())
    if pdf_sum > 0:
        pdf /= pdf_sum

    if remain > 0:
        free_idx = np.where(mask_1d == 0)[0]
        if free_idx.size > 0:
            probs = pdf[free_idx]
            probs_sum = float(probs.sum())
            if probs_sum <= 0:
                probs = np.ones_like(probs, dtype=np.float64) / len(probs)
            else:
                probs = probs / probs_sum
            picks = rng.choice(free_idx, size=min(remain, free_idx.size), replace=False, p=probs)
            mask_1d[picks] = 1.0

    return np.repeat(mask_1d[:, None], nx, axis=1).astype(np.float32)


class SenseForwardOp:
    def __init__(self, sens_maps: np.ndarray, mask: np.ndarray):
        self.sens = sens_maps.astype(np.complex64)
        self.mask = mask.astype(np.float32)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        coil_imgs = self.sens[:, None, :, :] * x[None, ...]
        return self.mask[None, None, :, :] * fft2c(coil_imgs)

    def adjoint(self, k: np.ndarray) -> np.ndarray:
        img = ifft2c(self.mask[None, None, :, :] * k)
        return np.sum(np.conj(self.sens)[:, None, :, :] * img, axis=0)


def estimate_lipschitz(op: SenseForwardOp, shape: Tuple[int, int, int], n_iter: int = 20) -> float:
    rng = np.random.default_rng(0)
    z = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    z = z.astype(np.complex64)
    z /= np.linalg.norm(z) + 1e-12

    for _ in range(n_iter):
        z = op.adjoint(op(z))
        nrm = np.linalg.norm(z) + 1e-12
        z /= nrm

    Az = op.adjoint(op(z))
    L = float(np.real(np.vdot(z, Az)))
    return max(L, 1e-8)


def soft_data_consistency(op: SenseForwardOp, x: np.ndarray, y: np.ndarray, d: float, exact: bool = False) -> np.ndarray:
    pred = op(x)
    if exact:
        k_new = np.where(op.mask[None, None, :, :] > 0, y, pred)
    else:
        k_new = pred + d * op.mask[None, None, :, :] * (y - pred)
    return op.adjoint(k_new) + (x - op.adjoint(pred))


def sampled_k_error(op: SenseForwardOp, x: np.ndarray, y: np.ndarray) -> float:
    pred = op(x)
    return float(np.linalg.norm((pred - y) * op.mask[None, None, :, :]) / (np.linalg.norm(y) + 1e-12))
