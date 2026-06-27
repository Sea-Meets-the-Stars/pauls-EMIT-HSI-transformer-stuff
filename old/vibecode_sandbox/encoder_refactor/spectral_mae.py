"""
Spectral Masked Autoencoder (GHG paper Sec. 4.1 style).

Band-wise masking, encoder on visible spectral tokens only (per spatial patch),
decoder on full band sequence with mask tokens, spatial–spectral sinusoidal PE,
reconstruction via linear head + fold.

Tensor layout: cube (N, C, H, W); tokens (N, G, C, D) before masking where G = patch grid size.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Optional

import torch
import torch.nn as nn

from model_definition import PreLNTransformerEncoder


@dataclass
class SpectralMAEConfig:
    """Configuration for SpectralMaskedAutoencoder (paper-style spectral MAE)."""

    spatial_size: tuple[int, int]
    in_channels: int
    patch_size: int
    embed_dim: int
    encoder_depth: int
    decoder_depth: int
    num_heads: int
    p_mask: float = 0.8
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    bands_per_group: int = 1
    lambda_min_nm: float = 400.0
    lambda_max_nm: float = 2500.0
    extra: dict[str, Any] = field(default_factory=dict)


def spectral_mae_config_to_dict(cfg: SpectralMAEConfig) -> dict[str, Any]:
    d = asdict(cfg)
    d["spatial_size"] = list(cfg.spatial_size)
    return d


def spectral_mae_config_from_dict(d: dict[str, Any]) -> SpectralMAEConfig:
    raw = dict(d)
    if "spatial_size" in raw:
        raw["spatial_size"] = tuple(int(x) for x in raw["spatial_size"])
    allowed = {f.name for f in SpectralMAEConfig.__dataclass_fields__.values()}
    raw = {k: v for k, v in raw.items() if k in allowed}
    return SpectralMAEConfig(**raw)


@dataclass
class SpectralMAEOutput:
    """Outputs from SpectralMaskedAutoencoder.forward."""

    cube_reconstruction: torch.Tensor
    """(N, C, H, W) reconstructed cube."""

    mask_band_indices: torch.Tensor
    """1D long tensor of masked band indices for this forward (shared batch-wide)."""

    tokens_decoder: Optional[torch.Tensor] = None
    """(N, G, C, D) optional decoder token sequence for probing."""

    aux: dict[str, Any] = field(default_factory=dict)


def sinusoidal_encoding_1d(
    positions: torch.Tensor,
    dim: int,
    *,
    max_period: float = 10000.0,
) -> torch.Tensor:
    """
    Args:
        positions: (L,) float or int positions (can be normalized).
        dim: output dimension (must be even).

    Returns:
        (L, dim) sinusoidal features.
    """
    if dim % 2 != 0:
        raise ValueError("sinusoidal_encoding_1d requires even dim")
    device = positions.device
    dtype = positions.dtype
    half = dim // 2
    positions = positions.float()
    div_term = torch.exp(torch.arange(0, half, device=device, dtype=torch.float32) * (-torch.log(torch.tensor(max_period, device=device)) / dim))
    angle = positions.unsqueeze(1) * div_term.unsqueeze(0)
    emb = torch.zeros(positions.shape[0], dim, device=device, dtype=dtype)
    emb[:, 0::2] = torch.sin(angle).to(dtype)
    emb[:, 1::2] = torch.cos(angle).to(dtype)
    return emb


class SpatialSpectralPositionalEncoding(nn.Module):
    """
    Eq. (5)–(10): E_pos = E_spatial(x,y) + E_spectral(λ), summed into embed_dim.

    Spatial: sinusoidal on patch grid indices (xg, yg).
    Spectral: λ scaled per paper then sinusoidal.
    """

    def __init__(
        self,
        num_patches_h: int,
        num_patches_w: int,
        num_bands: int,
        embed_dim: int,
        lambda_min_nm: float = 400.0,
        lambda_max_nm: float = 2500.0,
    ):
        super().__init__()
        self.gh = num_patches_h
        self.gw = num_patches_w
        self.num_bands = num_bands
        self.embed_dim = embed_dim
        self.lambda_min_nm = lambda_min_nm
        self.lambda_max_nm = lambda_max_nm

        third = embed_dim // 3
        d_spatial_x = (third // 2) * 2
        d_spatial_y = (third // 2) * 2
        d_spec = embed_dim - d_spatial_x - d_spatial_y
        if d_spec <= 0 or d_spec % 2 != 0:
            raise ValueError(f"embed_dim={embed_dim} cannot be split into even x/y/spec segments")

        self._d_x = d_spatial_x
        self._d_y = d_spatial_y
        self._d_spec = d_spec

        n_spatial = max(num_patches_h, num_patches_w)
        self.register_buffer("_n_spatial", torch.tensor(float(n_spatial)))

    def forward(self, wavelengths_nm: torch.Tensor) -> torch.Tensor:
        """
        Args:
            wavelengths_nm: (C,) center wavelength per band in nanometers.

        Returns:
            pe: (1, G, C, D) positional encodings to add to patch embeddings.
        """
        device = wavelengths_nm.device
        dtype = wavelengths_nm.dtype
        gh, gw = self.gh, self.gw
        G = gh * gw
        C = wavelengths_nm.shape[0]
        d = self.embed_dim

        pe = torch.zeros(1, G, C, d, device=device, dtype=dtype)

        # Patch order matches nn.Unfold: column-major within each row of patches (hxw grid).
        g_idx = torch.arange(G, device=device, dtype=torch.float32)
        xg = (g_idx % float(gw)).long()
        yg = (g_idx // float(gw)).long()

        ex = sinusoidal_encoding_1d(xg.float(), self._d_x)
        ey = sinusoidal_encoding_1d(yg.float(), self._d_y)

        lam = wavelengths_nm.float().clamp(self.lambda_min_nm, self.lambda_max_nm)
        lam_scaled = (lam - self.lambda_min_nm) / max(self.lambda_max_nm - self.lambda_min_nm, 1e-6) * self._n_spatial.to(
            device
        )
        es = sinusoidal_encoding_1d(lam_scaled, self._d_spec)

        pe[0, :, :, : self._d_x] = ex.unsqueeze(1).expand(-1, C, -1)
        pe[0, :, :, self._d_x : self._d_x + self._d_y] = ey.unsqueeze(1).expand(-1, C, -1)
        pe[0, :, :, self._d_x + self._d_y :] = es.unsqueeze(0).expand(G, -1, -1)

        return pe


class SpectralMaskedAutoencoder(nn.Module):
    """
    Spectral MAE: per spatial p×p patch, one token per spectral band (bands_per_group=1).

    Encoder input: only **visible** band tokens (flatten N*G × C_vis × D).
    Decoder input: **full** C bands with mask token at masked slots.
    """

    def __init__(self, cfg: SpectralMAEConfig):
        super().__init__()
        self.cfg = cfg
        h, w = cfg.spatial_size
        p = cfg.patch_size
        if h % p != 0 or w % p != 0:
            raise ValueError(f"spatial ({h},{w}) must divide patch_size {p}")
        c = cfg.in_channels
        if cfg.bands_per_group != 1:
            raise ValueError("bands_per_group > 1 not implemented in this version")
        self.patch_size = p
        self.in_channels = c
        self.gh = h // p
        self.gw = w // p
        self.num_patches = self.gh * self.gw
        self.embed_dim = cfg.embed_dim

        patch_dim = p * p
        self.patch_embed = nn.Linear(patch_dim, cfg.embed_dim)
        self.pos_encoding = SpatialSpectralPositionalEncoding(
            self.gh,
            self.gw,
            c,
            cfg.embed_dim,
            lambda_min_nm=cfg.lambda_min_nm,
            lambda_max_nm=cfg.lambda_max_nm,
        )

        self.encoder = PreLNTransformerEncoder(
            embed_dim=cfg.embed_dim,
            depth=cfg.encoder_depth,
            num_heads=cfg.num_heads,
            mlp_ratio=cfg.mlp_ratio,
            dropout=cfg.dropout,
        )
        self.decoder = PreLNTransformerEncoder(
            embed_dim=cfg.embed_dim,
            depth=cfg.decoder_depth,
            num_heads=cfg.num_heads,
            mlp_ratio=cfg.mlp_ratio,
            dropout=cfg.dropout,
        )

        self.mask_token = nn.Parameter(torch.zeros(1, 1, 1, cfg.embed_dim))
        nn.init.normal_(self.mask_token, std=0.02)

        self.decoder_pred = nn.Linear(cfg.embed_dim, patch_dim)

        self.register_buffer(
            "_default_wavelengths",
            torch.linspace(cfg.lambda_min_nm, cfg.lambda_max_nm, c),
            persistent=False,
        )

    def embed_patches(self, x: torch.Tensor) -> torch.Tensor:
        """(N,C,H,W) -> (N,G,C,D)"""
        n, c, h, w = x.shape
        p = self.patch_size
        unfold = nn.Unfold(kernel_size=p, stride=p)
        flat = unfold(x)
        g = flat.shape[2]
        flat = flat.transpose(1, 2).reshape(n, g, c, p * p)
        return self.patch_embed(flat)

    def forward(
        self,
        x: torch.Tensor,
        wavelengths_nm: Optional[torch.Tensor] = None,
        mask_band_indices: Optional[torch.Tensor] = None,
    ) -> SpectralMAEOutput:
        """
        Args:
            x: (N, C, H, W) normalized cube.
            wavelengths_nm: (C,) optional; defaults to uniform grid in [lambda_min, lambda_max].
            mask_band_indices: optional 1D long tensor of band indices to mask; if None, sample using p_mask.

        Returns:
            SpectralMAEOutput with full reconstructed cube (all bands predicted).
        """
        n, c, h, w = x.shape
        assert c == self.in_channels
        device = x.device
        dtype = x.dtype

        if wavelengths_nm is None:
            wavelengths_nm = self._default_wavelengths.to(device=device, dtype=torch.float32)

        tokens = self.embed_patches(x)
        pe = self.pos_encoding(wavelengths_nm.to(device=device, dtype=torch.float32))
        tokens = tokens + pe.to(dtype=dtype)

        if mask_band_indices is None:
            num_masked = int(round(self.cfg.p_mask * c))
            num_masked = max(1, min(c - 1, num_masked))
            perm = torch.randperm(c, device=device)
            mask_band_indices = perm[:num_masked].sort()[0]

        mask_set = torch.zeros(c, dtype=torch.bool, device=device)
        mask_set[mask_band_indices] = True
        visible_idx = torch.where(~mask_set)[0]

        g = self.num_patches
        enc_in = tokens[:, :, visible_idx, :].reshape(n * g, visible_idx.numel(), self.embed_dim)
        enc_out = self.encoder(enc_in)

        dec_full = self.mask_token.expand(n, g, c, self.embed_dim).clone()
        dec_full[:, :, visible_idx, :] = enc_out.reshape(n, g, visible_idx.numel(), self.embed_dim)

        dec_seq = dec_full.reshape(n * g, c, self.embed_dim)
        dec_seq = self.decoder(dec_seq)

        patch_flat = self.decoder_pred(dec_seq)
        p = self.patch_size
        patch_flat = patch_flat.reshape(n, g, c, p * p)
        patch_flat = patch_flat.transpose(1, 2).reshape(n, c * p * p, g)
        fold = nn.Fold(output_size=(h, w), kernel_size=p, stride=p)
        cube_hat = fold(patch_flat)

        return SpectralMAEOutput(
            cube_reconstruction=cube_hat,
            mask_band_indices=mask_band_indices.detach(),
            tokens_decoder=dec_full,
            aux={"visible_band_indices": visible_idx},
        )


def build_spectral_mae_model(cfg: SpectralMAEConfig) -> SpectralMaskedAutoencoder:
    """Validate and construct SpectralMaskedAutoencoder."""
    h, w = cfg.spatial_size
    if cfg.patch_size <= 0 or h % cfg.patch_size or w % cfg.patch_size:
        raise ValueError("invalid spatial_size or patch_size")
    if cfg.embed_dim % cfg.num_heads != 0:
        raise ValueError("embed_dim must be divisible by num_heads")
    if not (0.0 < cfg.p_mask < 1.0):
        raise ValueError("p_mask must be in (0,1)")
    return SpectralMaskedAutoencoder(cfg)
