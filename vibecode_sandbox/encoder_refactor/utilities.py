"""
Helpers for EMIT hyperspectral cubes indexed by dataset_index.csv.

Disk layout matches emit_utils.save_hypercube: arrays are (H, W, C); models use NCHW.
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Callable, Literal, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

_ENC_ROOT = Path(__file__).resolve().parent.parent

EMIT_FILL_VALUE = -9999.0


def _emit_utils():
    """Lazy import so importing ``utilities`` does not require EMIT optional deps."""
    import sys

    if str(_ENC_ROOT) not in sys.path:
        sys.path.insert(0, str(_ENC_ROOT))
    import emit_utils

    return emit_utils

TensorLayout = Literal["NCHW", "HWC"]


@dataclass(frozen=True)
class TensorSpec:
    """Human-readable tensor shape contract for documentation and debugging."""

    layout: TensorLayout
    shape: tuple[Any, ...]
    description: str

    def __repr__(self) -> str:
        return f"TensorSpec({self.layout}, {self.shape}, {self.description!r})"


# --- Paths & index -----------------------------------------------------------


def resolve_hypercube_path(
    dataset_root: str | Path,
    event_id: str,
    prefer_fmt: Optional[Literal["npy", "hdf5"]] = None,
) -> tuple[str, Literal["npy", "hdf5"]]:
    """
    Resolve path to a chipped hypercube file under dataset_root / event_id.

    Args:
        dataset_root: Root directory passed to emit_utils.build_dataset (contains dataset_index.csv).
        event_id: Relative directory from CSV (e.g. '20230825T163454_ch4/plume_id_easy').
        prefer_fmt: If set, try this extension first.

    Returns:
        (absolute_path, fmt) where fmt is 'npy' or 'hdf5'.

    Raises:
        FileNotFoundError if neither hypercube.npy nor hypercube.h5 exists.
    """
    root = Path(dataset_root)
    base = root / event_id
    candidates: list[tuple[str, Literal["npy", "hdf5"]]] = []
    if prefer_fmt == "npy":
        candidates = [(str(base / "hypercube.npy"), "npy"), (str(base / "hypercube.h5"), "hdf5")]
    elif prefer_fmt == "hdf5":
        candidates = [(str(base / "hypercube.h5"), "hdf5"), (str(base / "hypercube.npy"), "npy")]
    else:
        candidates = [
            (str(base / "hypercube.npy"), "npy"),
            (str(base / "hypercube.h5"), "hdf5"),
        ]

    for path_str, fmt in candidates:
        if os.path.isfile(path_str):
            return path_str, fmt

    raise FileNotFoundError(f"No hypercube.npy or hypercube.h5 under {base}")


def load_dataset_index_csv(csv_path: str | Path) -> pd.DataFrame:
    """Load dataset_index.csv produced by emit_utils.build_dataset."""
    return pd.read_csv(csv_path)


def filter_index_rows(
    df: pd.DataFrame,
    *,
    labels: Optional[list[str]] = None,
    gas_types: Optional[list[str]] = None,
    training_categories: Optional[list[str]] = None,
) -> pd.DataFrame:
    """Filter rows by optional column subsets (None = no filter on that field)."""
    out = df.copy()
    if labels is not None:
        out = out[out["label"].isin(labels)]
    if gas_types is not None and "gas_type" in out.columns:
        out = out[out["gas_type"].isin(gas_types)]
    if training_categories is not None and "training_category" in out.columns:
        in_cat = out["training_category"].isin(training_categories)
        is_neg = out["label"].eq("negative")
        out = out[in_cat | is_neg]
    return out.reset_index(drop=True)


def train_val_split_by_granule(
    df: pd.DataFrame,
    val_fraction: float,
    seed: int,
    granule_column: str = "granule_id",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split rows so all chips from the same granule stay in one split (no leakage).
    """
    rng = np.random.default_rng(seed)
    granules = df[granule_column].drop_duplicates().tolist()
    rng.shuffle(granules)
    n_val = max(1, int(len(granules) * val_fraction)) if len(granules) > 1 else 1
    val_set = set(granules[:n_val])
    val_mask = df[granule_column].isin(val_set)
    return df[~val_mask].reset_index(drop=True), df[val_mask].reset_index(drop=True)


def load_instrument_metadata(dataset_dir: str | Path) -> dict[str, Any]:
    """Load instrument.json (wavelengths, fwhm, num_bands)."""
    path = Path(dataset_dir) / "instrument.json"
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def num_bands_from_instrument(dataset_dir: str | Path) -> int:
    meta = load_instrument_metadata(dataset_dir)
    return int(meta.get("num_bands", len(meta["wavelengths"])))


def load_wavelengths_nm_from_instrument(dataset_dir: str | Path) -> np.ndarray:
    """Center wavelengths per band from instrument.json (nanometers)."""
    meta = load_instrument_metadata(dataset_dir)
    return np.asarray(meta["wavelengths"], dtype=np.float32)


# --- I/O ---------------------------------------------------------------------


def load_hypercube_chip(path: str, fmt: Optional[Literal["npy", "hdf5"]] = None) -> np.ndarray:
    """
    Load one chip; returns (H, W, C) float array (mmap for npy when possible).

    Delegates to emit_utils.load_hypercube.
    """
    return _emit_utils().load_hypercube(path, fmt=fmt)


# --- Transforms --------------------------------------------------------------


def hwc_numpy_to_chw_float(
    arr: np.ndarray,
    *,
    dtype: np.dtype = np.float32,
) -> torch.Tensor:
    """
    Args:
        arr: (H, W, C) numpy array.

    Returns:
        (C, H, W) float32 tensor.
    """
    if arr.dtype != dtype:
        arr = arr.astype(dtype, copy=False)
    t = torch.from_numpy(np.asarray(arr))
    if t.ndim != 3:
        raise ValueError(f"Expected HWC array, got shape {tuple(t.shape)}")
    return t.permute(2, 0, 1).contiguous()


def normalize_cube_chw(
    x_chw: torch.Tensor,
    band_mean: torch.Tensor,
    band_std: torch.Tensor,
    fill_value: float = EMIT_FILL_VALUE,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Per-band z-score on valid pixels; invalid pixels set to 0 after normalization.

    Args:
        x_chw: (C, H, W)
        band_mean: (C,) or (C, 1, 1)
        band_std: (C,) or (C, 1, 1)

    Returns:
        normalized (C, H, W), valid mask (C, H, W) float {0,1}
    """
    mean = band_mean.view(-1, 1, 1)
    std = band_std.view(-1, 1, 1)
    valid = (x_chw != fill_value) & torch.isfinite(x_chw)
    safe_std = std.clamp(min=1e-6)
    out = torch.where(valid, (x_chw - mean) / safe_std, torch.zeros_like(x_chw))
    return out, valid.float()


def load_band_stats_npz(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Load band_mean and band_std from an .npz (same keys as hsi_models save_band_stats).
    """
    d = np.load(path)
    return d["band_mean"].astype(np.float32), d["band_std"].astype(np.float32)


# --- Dataset -----------------------------------------------------------------


class EmitHypercubeIndexDataset(Dataset):
    """
    PyTorch Dataset over rows of dataset_index.csv.

    Each item is a dict:
      - ``cube``: float tensor (C, H, W) NCHW layout (single-sample CHW; collate stacks batch dim).
      - ``valid``: float mask (C, H, W)
      - ``meta``: row metadata (id, label, granule_id, event_id, ...)

    ``cube`` is optionally normalized if ``band_mean`` and ``band_std`` are provided.
    """

    def __init__(
        self,
        index_df: pd.DataFrame,
        dataset_root: str | Path,
        *,
        band_mean: Optional[np.ndarray] = None,
        band_std: Optional[np.ndarray] = None,
        fill_value: float = EMIT_FILL_VALUE,
        prefer_fmt: Optional[Literal["npy", "hdf5"]] = None,
        transform: Optional[Callable[[dict[str, Any]], dict[str, Any]]] = None,
    ):
        self.index_df = index_df.reset_index(drop=True)
        self.dataset_root = Path(dataset_root)
        self.fill_value = fill_value
        self.prefer_fmt = prefer_fmt
        self.transform = transform

        if band_mean is not None and band_std is not None:
            self.band_mean = torch.tensor(band_mean, dtype=torch.float32)
            self.band_std = torch.tensor(band_std, dtype=torch.float32)
        else:
            self.band_mean = None
            self.band_std = None

    def __len__(self) -> int:
        return len(self.index_df)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.index_df.iloc[idx]
        event_id = row["event_id"]
        path, fmt = resolve_hypercube_path(self.dataset_root, event_id, prefer_fmt=self.prefer_fmt)
        arr = load_hypercube_chip(path, fmt=fmt)
        x = hwc_numpy_to_chw_float(arr, dtype=np.float32)

        if self.band_mean is not None and self.band_std is not None:
            x, valid = normalize_cube_chw(x, self.band_mean, self.band_std, fill_value=self.fill_value)
        else:
            valid = (x != self.fill_value) & torch.isfinite(x)
            valid = valid.float()

        meta = row.to_dict()
        sample: dict[str, Any] = {"cube": x, "valid": valid, "meta": meta}
        if self.transform is not None:
            sample = self.transform(sample)
        return sample


def collate_hypercube_batch(samples: list[dict[str, Any]]) -> dict[str, Any]:
    """Stack single-chip dicts into batch tensors."""
    cubes = torch.stack([s["cube"] for s in samples], dim=0)
    valids = torch.stack([s["valid"] for s in samples], dim=0)
    metas = [s["meta"] for s in samples]
    return {"cube": cubes, "valid": valids, "meta": metas}


# --- Training helpers --------------------------------------------------------


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_checkpoint(
    path: str | Path,
    *,
    model_state_dict: dict[str, Any],
    model_config_dict: dict[str, Any],
    optimizer_state_dict: Optional[dict[str, Any]] = None,
    scheduler_state_dict: Optional[dict[str, Any]] = None,
    epoch: Optional[int] = None,
    extra: Optional[dict[str, Any]] = None,
) -> None:
    """Torch checkpoint with JSON-serializable model_config_dict."""
    payload: dict[str, Any] = {
        "model_state_dict": model_state_dict,
        "model_config": model_config_dict,
    }
    if optimizer_state_dict is not None:
        payload["optimizer_state_dict"] = optimizer_state_dict
    if scheduler_state_dict is not None:
        payload["scheduler_state_dict"] = scheduler_state_dict
    if epoch is not None:
        payload["epoch"] = epoch
    if extra:
        payload["extra"] = extra
    torch.save(payload, path)


def load_checkpoint(path: str | Path, map_location: str | torch.device | None = None) -> dict[str, Any]:
    """Load checkpoint dict from save_checkpoint."""
    return torch.load(path, map_location=map_location, weights_only=False)


# --- Loss (self-contained; matches masked_band_mse semantics from hsi_models) ---


def masked_mse(x_hat: torch.Tensor, x: torch.Tensor, valid: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Pixel MSE weighted by valid mask; tensors (B, C, H, W)."""
    diff2 = (x_hat - x) ** 2 * valid
    return diff2.sum() / valid.sum().clamp_min(eps)


def masked_band_mse(x_hat: torch.Tensor, x: torch.Tensor, valid: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Average over bands of per-band masked MSE; tensors (B, C, H, W)."""
    diff2 = (x_hat - x) ** 2 * valid
    denom_per_band = valid.sum(dim=(0, 2, 3)).clamp_min(eps)
    mse_per_band = diff2.sum(dim=(0, 2, 3)) / denom_per_band
    return mse_per_band.mean()


def masked_l1_full_cube(
    x_hat: torch.Tensor,
    x: torch.Tensor,
    valid: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Mean absolute error averaged over valid pixels (paper Eq. 15 style)."""
    diff = (x_hat - x).abs() * valid
    return diff.sum() / valid.sum().clamp_min(eps)


def band_masked_l1(
    x_hat: torch.Tensor,
    x: torch.Tensor,
    valid: torch.Tensor,
    band_mask: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    MAE averaged over valid pixels whose spectral band is selected in ``band_mask``.

    Args:
        band_mask: (C,) bool; True for bands included in the loss (e.g. MAE-masked bands).
    """
    w = band_mask.float().view(1, -1, 1, 1).to(device=x.device, dtype=x.dtype)
    diff = (x_hat - x).abs() * valid * w
    denom = (valid * w).sum().clamp_min(eps)
    return diff.sum() / denom


def band_indices_to_mask(num_bands: int, indices: torch.Tensor, *, device: torch.device) -> torch.Tensor:
    """1D long tensor of band indices -> (C,) bool with True at ``indices``."""
    m = torch.zeros(num_bands, dtype=torch.bool, device=device)
    m[indices.long()] = True
    return m


def linear_warmup_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
    min_lr_ratio: float = 0.0,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Call ``scheduler.step()`` once per batch; linear warmup then cosine decay."""

    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return float(current_step + 1) / float(max(1, warmup_steps))
        progress = (current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
