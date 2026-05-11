#!/usr/bin/env python3
"""Minimal spectral MAE on EMIT hypercube chips. Requires: torch, numpy."""

from __future__ import annotations

import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# --- knobs ---
DATA_ROOT = Path.home() / "emit_data"
PATCH_SIZE = 16
P_MASK = 0.8
BATCH = 1  # large H,W + full-spectrum decoder needs small batch on GPU
EPOCHS = 50
LR = 1e-4
AUGMENT = True
AUG_H_FLIP_P = 0.5
AUG_V_FLIP_P = 0.5
EMBED_DIM = 256
ENC_DEPTH = 4
DEC_DEPTH = 4
NUM_HEADS = 8
MLP_RATIO = 4.0
DROPOUT = 0.0
LAMBDA_MIN_NM = 400.0
LAMBDA_MAX_NM = 2500.0
LOG_EVERY = 10
# Joint objective: masked MAE + weight on full-cube MAE (visible bands otherwise get no gradient).
LOSS_ALPHA_FULL = 1.0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _sinu_1d(pos: torch.Tensor, dim: int, max_period: float = 10000.0) -> torch.Tensor:
    if dim % 2:
        raise ValueError("sinusoidal dim must be even")
    device = pos.device
    half = dim // 2
    pos = pos.float()
    div_term = torch.exp(
        torch.arange(0, half, device=device, dtype=torch.float32)
        * (-torch.log(torch.tensor(max_period, device=device)) / dim)
    )
    ang = pos.unsqueeze(1) * div_term.unsqueeze(0)
    out = torch.zeros(pos.shape[0], dim, device=device, dtype=pos.dtype)
    out[:, 0::2] = torch.sin(ang).to(pos.dtype)
    out[:, 1::2] = torch.cos(ang).to(pos.dtype)
    return out


def list_hypercube_paths(root: Path) -> list[Path]:
    if not root.is_dir():
        raise FileNotFoundError(f"DATA_ROOT does not exist: {root}")
    paths = sorted(root.rglob("hypercube.npy"))
    out = [p for p in paths if not any(part.startswith("_") for part in p.parts)]
    if not out:
        raise ValueError(f"No hypercube.npy under {root} (after skipping _* dirs)")
    return out


def load_instrument(root: Path) -> tuple[np.ndarray, int]:
    p = root / "instrument.json"
    if not p.is_file():
        raise FileNotFoundError(f"Missing {p}")
    with open(p, encoding="utf-8") as f:
        meta = json.load(f)
    wl = np.asarray(meta["wavelengths"], dtype=np.float32)
    nb = int(meta["num_bands"])
    if wl.shape[0] != nb:
        raise ValueError("instrument.json num_bands != len(wavelengths)")
    return wl, nb


def compute_band_stats(paths: list[Path], num_bands: int) -> tuple[np.ndarray, np.ndarray]:
    """Single pass: validate every chip and accumulate global band mean/variance."""
    sum_b = np.zeros(num_bands, dtype=np.float64)
    sumsq_b = np.zeros(num_bands, dtype=np.float64)
    n_pix = 0
    bad: list[str] = []
    for fp in paths:
        arr = np.load(fp, mmap_mode="r")
        if arr.ndim != 3 or arr.shape[2] != num_bands:
            bad.append(f"{fp} shape {arr.shape} (expected (*,*,{num_bands}))")
            continue
        if not np.isfinite(arr).all():
            bad.append(f"{fp} contains NaN or Inf")
            continue
        x = np.asarray(arr, dtype=np.float64)
        sum_b += x.reshape(-1, num_bands).sum(axis=0)
        sumsq_b += np.square(x.reshape(-1, num_bands)).sum(axis=0)
        n_pix += arr.shape[0] * arr.shape[1]
    if bad:
        raise ValueError("Invalid hypercube files — fix or remove:\n" + "\n".join(bad))
    mean64 = sum_b / n_pix
    var = sumsq_b / n_pix - mean64**2
    if np.any(var < 0) or not np.isfinite(var).all():
        raise ValueError("Non-finite or negative variance — check data")
    mean = mean64.astype(np.float32)
    std = np.sqrt(var).astype(np.float32)
    if np.any(std <= 0):
        raise ValueError("Zero std for at least one band — check data")
    return mean, std


def augment_spatial_hyperspectral(x: torch.Tensor) -> torch.Tensor:
    """
    Random H/V flips and k·90° rotation in the plane (dims H,W). Requires square H==W.
    Operates on normalized (C,H,W) tensors.
    """
    c, h, w = x.shape
    if h != w:
        raise ValueError(f"Spatial augmentation needs square chips, got H={h} W={w}")
    k = random.randint(0, 3)
    if k:
        x = torch.rot90(x, k=k, dims=(1, 2))
    if random.random() < AUG_H_FLIP_P:
        x = torch.flip(x, dims=(2,))
    if random.random() < AUG_V_FLIP_P:
        x = torch.flip(x, dims=(1,))
    return x.contiguous()


class HypercubeDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        paths: list[Path],
        band_mean: np.ndarray,
        band_std: np.ndarray,
        augment: bool,
        spatial_size: tuple[int, int],
    ):
        self.paths = paths
        self.band_mean = torch.from_numpy(band_mean).float().view(-1, 1, 1)
        self.band_std = torch.from_numpy(band_std).float().view(-1, 1, 1)
        self.augment = augment
        self._h, self._w = spatial_size

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i: int) -> torch.Tensor:
        arr = np.load(self.paths[i], mmap_mode="r")
        x = torch.from_numpy(np.array(arr, dtype=np.float32, copy=True)).permute(2, 0, 1)
        x = (x - self.band_mean) / self.band_std
        if self.augment:
            if x.shape[1] != self._h or x.shape[2] != self._w:
                raise ValueError(f"Chip spatial shape {x.shape[1:]} != dataset ({self._h},{self._w})")
            x = augment_spatial_hyperspectral(x)
        return x


class SpatialSpectralPE(nn.Module):
    def __init__(self, gh: int, gw: int, embed_dim: int):
        super().__init__()
        third = embed_dim // 3
        dx = (third // 2) * 2
        dy = (third // 2) * 2
        ds = embed_dim - dx - dy
        if ds <= 0 or ds % 2:
            raise ValueError("embed_dim cannot be split for PE")
        self.gh, self.gw = gh, gw
        self.dx, self.dy, self.ds = dx, dy, ds
        self.embed_dim = embed_dim
        self.register_buffer("_n_sp", torch.tensor(float(max(gh, gw))))

    def forward(self, wl_nm: torch.Tensor) -> torch.Tensor:
        G = self.gh * self.gw
        C = wl_nm.shape[0]
        d = self.embed_dim
        device, dtype = wl_nm.device, wl_nm.dtype
        pe = torch.zeros(1, G, C, d, device=device, dtype=dtype)
        g_idx = torch.arange(G, device=device, dtype=torch.float32)
        xg = (g_idx % float(self.gw)).long()
        yg = (g_idx // float(self.gw)).long()
        ex = _sinu_1d(xg.float(), self.dx)
        ey = _sinu_1d(yg.float(), self.dy)
        span = LAMBDA_MAX_NM - LAMBDA_MIN_NM
        if span <= 0:
            raise ValueError("LAMBDA_MAX_NM must exceed LAMBDA_MIN_NM")
        lam = wl_nm.float().clamp(LAMBDA_MIN_NM, LAMBDA_MAX_NM)
        lam_s = (lam - LAMBDA_MIN_NM) / span * self._n_sp.to(device)
        es = _sinu_1d(lam_s, self.ds)
        pe[0, :, :, : self.dx] = ex.unsqueeze(1).expand(-1, C, -1)
        pe[0, :, :, self.dx : self.dx + self.dy] = ey.unsqueeze(1).expand(-1, C, -1)
        pe[0, :, :, self.dx + self.dy :] = es.unsqueeze(0).expand(G, -1, -1)
        return pe


class SpectralMAE(nn.Module):
    def __init__(self, c: int, h: int, w: int, patch: int, dim: int, enc_d: int, dec_d: int, heads: int):
        super().__init__()
        if h % patch or w % patch:
            raise ValueError("H,W must be divisible by patch_size")
        if dim % heads:
            raise ValueError("embed_dim must divide num_heads")
        self.c, self.h, self.w, self.p = c, h, w, patch
        self.gh, self.gw = h // patch, w // patch
        self.g = self.gh * self.gw
        self.dim = dim
        pd = patch * patch
        self.patch_embed = nn.Linear(pd, dim)
        self.pos = SpatialSpectralPE(self.gh, self.gw, dim)
        el = nn.TransformerEncoderLayer(
            dim, heads, int(dim * MLP_RATIO), DROPOUT, batch_first=True, activation="gelu"
        )
        dl = nn.TransformerEncoderLayer(
            dim, heads, int(dim * MLP_RATIO), DROPOUT, batch_first=True, activation="gelu"
        )
        self.encoder = nn.TransformerEncoder(el, enc_d)
        self.decoder = nn.TransformerEncoder(dl, dec_d)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, 1, dim))
        nn.init.normal_(self.mask_token, std=0.02)
        self.head = nn.Linear(dim, pd)
        self.unfold = nn.Unfold(kernel_size=patch, stride=patch)
        self.fold = nn.Fold(output_size=(h, w), kernel_size=patch, stride=patch)

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        n, c, h, w = x.shape
        flat = self.unfold(x)
        g = flat.shape[2]
        tok = flat.transpose(1, 2).reshape(n, g, c, self.p * self.p)
        return self.patch_embed(tok)

    def forward(self, x: torch.Tensor, wl: torch.Tensor, mask_idx: torch.Tensor | None):
        n, c, h, w = x.shape
        assert c == self.c and h == self.h and w == self.w
        tokens = self.embed(x)
        pe = self.pos(wl.to(device=x.device, dtype=x.dtype))
        tokens = tokens + pe.to(dtype=x.dtype)
        device = x.device
        if mask_idx is None:
            nm = int(round(P_MASK * c))
            nm = max(1, min(c - 1, nm))
            perm = torch.randperm(c, device=device)
            mask_idx = perm[:nm].sort()[0]
        ms = torch.zeros(c, dtype=torch.bool, device=device)
        ms[mask_idx] = True
        vis = torch.where(~ms)[0]
        enc_in = tokens[:, :, vis, :].reshape(n * self.g, vis.numel(), self.dim)
        enc_out = self.encoder(enc_in)
        dec = self.mask_token.expand(n, self.g, c, self.dim).clone()
        dec[:, :, vis, :] = enc_out.reshape(n, self.g, vis.numel(), self.dim)
        dec_seq = self.decoder(dec.reshape(n * self.g, c, self.dim))
        pf = self.head(dec_seq).reshape(n, self.g, c, self.p * self.p)
        pf = pf.transpose(1, 2).reshape(n, c * self.p * self.p, self.g)
        return self.fold(pf), mask_idx


def main():
    epochs = int(os.environ["MAE_EPOCHS"]) if "MAE_EPOCHS" in os.environ else EPOCHS
    paths = list_hypercube_paths(DATA_ROOT)
    wl_np, num_bands = load_instrument(DATA_ROOT)
    mean, std = compute_band_stats(paths, num_bands)
    wl = torch.from_numpy(wl_np).to(DEVICE)

    # infer H,W from first file (all chips must match)
    z = np.load(paths[0], mmap_mode="r")
    H, W, C = z.shape
    if C != num_bands:
        raise ValueError("chip bands != instrument num_bands")
    if H % PATCH_SIZE or W % PATCH_SIZE:
        raise ValueError("chip H,W must divide PATCH_SIZE")

    ds = HypercubeDataset(paths, mean, std, augment=AUGMENT, spatial_size=(H, W))
    loader = torch.utils.data.DataLoader(ds, batch_size=BATCH, shuffle=True, num_workers=0, drop_last=True)

    model = SpectralMAE(C, H, W, PATCH_SIZE, EMBED_DIM, ENC_DEPTH, DEC_DEPTH, NUM_HEADS).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    step = 0
    for ep in range(epochs):
        for batch in loader:
            batch = batch.to(DEVICE)
            pred, midx = model(batch, wl, None)
            m = torch.zeros(C, dtype=torch.bool, device=DEVICE)
            m[midx] = True
            abs_err = (pred - batch).abs()
            loss_masked = abs_err[:, m].mean()
            loss_full = abs_err.mean()
            loss = loss_masked + LOSS_ALPHA_FULL * loss_full
            opt.zero_grad()
            loss.backward()
            opt.step()
            step += 1
            if step % LOG_EVERY == 0:
                print(
                    f"epoch {ep+1}/{epochs} step {step} "
                    f"loss {loss.item():.6f} mae_masked {loss_masked.item():.6f} mae_full {loss_full.item():.6f}"
                )

    ckpt_path = Path(__file__).resolve().parent / "checkpoint.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "band_mean": torch.from_numpy(mean),
            "band_std": torch.from_numpy(std),
            "wavelengths": wl.cpu(),
            "meta": {
                "PATCH_SIZE": PATCH_SIZE,
                "H": H,
                "W": W,
                "C": C,
                "LOSS_ALPHA_FULL": LOSS_ALPHA_FULL,
                "epochs_ran": epochs,
            },
        },
        ckpt_path,
    )
    print(f"saved {ckpt_path}")


if __name__ == "__main__":
    main()
