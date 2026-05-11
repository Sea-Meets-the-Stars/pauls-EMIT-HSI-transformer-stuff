"""Reusable helpers for spectral MAE inspection (numpy RGB views + tensor I/O)."""

from __future__ import annotations

import numpy as np
import torch


def nearest_band_indices(
    wavelengths_nm: np.ndarray,
    targets_nm: tuple[float, ...] = (650.0, 550.0, 450.0),
) -> tuple[int, ...]:
    """Band indices closest to each target wavelength (e.g. fake RGB)."""
    return tuple(int(np.argmin(np.abs(wavelengths_nm - t))) for t in targets_nm)


def rgb_percentile_stretch(img_hwc: np.ndarray, p_lo: float = 2.0, p_hi: float = 98.0) -> np.ndarray:
    """Per-channel min/max via percentiles, clipped to [0, 1]."""
    x = np.zeros_like(img_hwc)
    for k in range(img_hwc.shape[-1]):
        lo, hi = np.percentile(img_hwc[..., k], (p_lo, p_hi))
        if hi <= lo:
            x[..., k] = 0.0
        else:
            x[..., k] = np.clip((img_hwc[..., k] - lo) / (hi - lo), 0.0, 1.0)
    return x


def tensor_hypercube_to_model_input(
    raw_hwc: np.ndarray,
    band_mean: torch.Tensor,
    band_std: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """``raw_hwc`` (H,W,C) float reflectance -> normalized (1,C,H,W) on ``device``."""
    x_cpu = torch.from_numpy(np.asarray(raw_hwc, dtype=np.float32, copy=True)).permute(2, 0, 1)
    bm = band_mean.detach().cpu().view(-1, 1, 1)
    bs = band_std.detach().cpu().view(-1, 1, 1)
    return ((x_cpu - bm) / bs).unsqueeze(0).to(device)


def denormalize_nchw(x: torch.Tensor, band_mean: torch.Tensor, band_std: torch.Tensor) -> torch.Tensor:
    """Invert band-wise z-score. ``x`` (N,C,H,W); ``band_mean`` / ``band_std`` broadcastable to (N,C,H,W)."""
    return x * band_std + band_mean
