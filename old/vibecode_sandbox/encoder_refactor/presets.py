"""
Named model presets for ViT reconstruction baselines and GHG-paper Spectral MAE.

Use ``build_spectral_mae_preset`` / ``build_vit_preset`` to merge defaults with overrides
(e.g. ``in_channels`` and ``spatial_size`` from your dataset).
"""

from __future__ import annotations

from typing import Any, Callable

from model_definition import ModelConfig
from spectral_mae import SpectralMAEConfig

SPECTRAL_MAE_PRESET_REGISTRY: dict[str, Callable[..., SpectralMAEConfig]] = {}
VIT_PRESET_REGISTRY: dict[str, Callable[..., ModelConfig]] = {}


def register_spectral_mae_preset(name: str) -> Callable[[Callable[..., SpectralMAEConfig]], Callable[..., SpectralMAEConfig]]:
    def deco(fn: Callable[..., SpectralMAEConfig]) -> Callable[..., SpectralMAEConfig]:
        SPECTRAL_MAE_PRESET_REGISTRY[name] = fn
        return fn

    return deco


def register_vit_preset(name: str) -> Callable[[Callable[..., ModelConfig]], Callable[..., ModelConfig]]:
    def deco(fn: Callable[..., ModelConfig]) -> Callable[..., ModelConfig]:
        VIT_PRESET_REGISTRY[name] = fn
        return fn

    return deco


@register_spectral_mae_preset("ghg_paper_spectral_mae")
def preset_ghg_paper_spectral_mae(
    *,
    spatial_size: tuple[int, int] = (128, 128),
    in_channels: int = 202,
    patch_size: int = 16,
    embed_dim: int = 256,
    encoder_depth: int = 12,
    decoder_depth: int = 4,
    num_heads: int = 8,
    p_mask: float = 0.8,
    mlp_ratio: float = 4.0,
    dropout: float = 0.0,
    lambda_min_nm: float = 400.0,
    lambda_max_nm: float = 2500.0,
    **kwargs: Any,
) -> SpectralMAEConfig:
    """
    Defaults aligned with the GHG Spectral Transformer MAE narrative (Sec. 4.1, 80% pre-training mask).

    Hyperparameters (depth, width) are placeholders — tune for compute budget; ``in_channels``
    should match ``instrument.json`` (EnMAP ~202 usable bands in the paper; EMIT differs).
    """
    kw = dict(
        spatial_size=spatial_size,
        in_channels=in_channels,
        patch_size=patch_size,
        embed_dim=embed_dim,
        encoder_depth=encoder_depth,
        decoder_depth=decoder_depth,
        num_heads=num_heads,
        p_mask=p_mask,
        mlp_ratio=mlp_ratio,
        dropout=dropout,
        lambda_min_nm=lambda_min_nm,
        lambda_max_nm=lambda_max_nm,
        bands_per_group=1,
    )
    kw.update(kwargs)
    return SpectralMAEConfig(**kw)


@register_vit_preset("vit_patch16_emit256")
def preset_vit_patch16_emit256(
    *,
    spatial_size: tuple[int, int] = (256, 256),
    in_channels: int = 285,
    embed_dim: int = 128,
    depth: int = 4,
    num_heads: int = 8,
    patch_size: int = 16,
    **kwargs: Any,
) -> ModelConfig:
    """Spatial-patch ViT autoencoder baseline (matches earlier encoder_refactor notebook defaults)."""
    kw = dict(
        spatial_size=spatial_size,
        in_channels=in_channels,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        mlp_ratio=4.0,
        dropout=0.0,
        tokenizer_type="patch2d",
        tokenizer_kwargs={"patch_h": patch_size, "patch_w": patch_size},
        pos_encoding_type="learned",
        backbone_type="preln_vit",
        head_type="cube_fold_patch2d",
        head_kwargs={},
        use_cls_token=False,
    )
    kw.update(kwargs)
    return ModelConfig(**kw)


def build_spectral_mae_preset(name: str, **overrides: Any) -> SpectralMAEConfig:
    if name not in SPECTRAL_MAE_PRESET_REGISTRY:
        raise KeyError(f"Unknown Spectral MAE preset {name!r}; options: {list(SPECTRAL_MAE_PRESET_REGISTRY)}")
    return SPECTRAL_MAE_PRESET_REGISTRY[name](**overrides)


def build_vit_preset(name: str, **overrides: Any) -> ModelConfig:
    if name not in VIT_PRESET_REGISTRY:
        raise KeyError(f"Unknown ViT preset {name!r}; options: {list(VIT_PRESET_REGISTRY)}")
    return VIT_PRESET_REGISTRY[name](**overrides)
