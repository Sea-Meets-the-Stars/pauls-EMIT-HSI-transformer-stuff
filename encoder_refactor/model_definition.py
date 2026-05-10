"""
Composable ViT-style encoder for hyperspectral cubes (NCHW).

Swap tokenizers, positional encoding, backbone, and reconstruction head via ModelConfig + registries.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Optional

import torch
import torch.nn as nn
from utilities import TensorSpec


# --- Specs -------------------------------------------------------------------


@dataclass(frozen=True)
class ComponentSpec:
    """Declared I/O for wiring validation."""

    name: str
    in_spec: TensorSpec
    out_spec: TensorSpec
    embed_dim: int
    num_tokens: int


@dataclass
class ModelOutput:
    """Forward pass return bundle for losses and logging."""

    tokens_encoded: torch.Tensor
    """Shape (B, N, D) encoder output sequence (after backbone)."""

    cube_reconstruction: Optional[torch.Tensor] = None
    """Shape (B, C, H, W) if head predicts full cube."""

    aux: dict[str, Any] = field(default_factory=dict)


# --- Model config ------------------------------------------------------------


@dataclass
class ModelConfig:
    """Single source of truth for spatial size, channels, and component choices."""

    spatial_size: tuple[int, int]
    in_channels: int
    embed_dim: int
    depth: int
    num_heads: int
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    tokenizer_type: str = "patch2d"
    tokenizer_kwargs: dict[str, Any] = field(default_factory=dict)
    pos_encoding_type: str = "learned"
    pos_encoding_kwargs: dict[str, Any] = field(default_factory=dict)
    backbone_type: str = "preln_vit"
    backbone_kwargs: dict[str, Any] = field(default_factory=dict)
    head_type: str = "cube_fold_patch2d"
    head_kwargs: dict[str, Any] = field(default_factory=dict)
    use_cls_token: bool = False


def model_config_to_dict(cfg: ModelConfig) -> dict[str, Any]:
    d = asdict(cfg)
    d["spatial_size"] = list(cfg.spatial_size)
    return d


def model_config_from_dict(d: dict[str, Any]) -> ModelConfig:
    raw = dict(d)
    if "spatial_size" in raw:
        raw["spatial_size"] = tuple(int(x) for x in raw["spatial_size"])
    return ModelConfig(**raw)


# --- Registries --------------------------------------------------------------


TOKENIZER_REGISTRY: dict[str, type["HyperspectralTokenizer"]] = {}
POS_REGISTRY: dict[str, type["PositionalEncoding"]] = {}
BACKBONE_REGISTRY: dict[str, type["TransformerEncoderBackbone"]] = {}
HEAD_REGISTRY: dict[str, type["PredictionHead"]] = {}


def register_tokenizer(name: str) -> Callable[[type], type]:
    def deco(cls: type) -> type:
        TOKENIZER_REGISTRY[name] = cls
        return cls

    return deco


def register_pos(name: str) -> Callable[[type], type]:
    def deco(cls: type) -> type:
        POS_REGISTRY[name] = cls
        return cls

    return deco


def register_backbone(name: str) -> Callable[[type], type]:
    def deco(cls: type) -> type:
        BACKBONE_REGISTRY[name] = cls
        return cls

    return deco


def register_head(name: str) -> Callable[[type], type]:
    def deco(cls: type) -> type:
        HEAD_REGISTRY[name] = cls
        return cls

    return deco


# --- Abstract components -----------------------------------------------------


class HyperspectralTokenizer(nn.Module, ABC):
    """
    Maps input cube (B, C, H, W) to patch tokens (B, N, D).

    Subclasses must set ``embed_dim``, ``num_tokens``, ``patch_volume`` (flattened values per token).
    """

    embed_dim: int
    num_tokens: int
    patch_volume: int

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args: x (B, C, H, W). Returns: tokens (B, N, D)."""


class PositionalEncoding(nn.Module, ABC):
    @abstractmethod
    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Args: tokens (B, N, D). Returns: same shape with position info added."""


class TransformerEncoderBackbone(nn.Module, ABC):
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args: x (B, N, D). Returns: (B, N, D)."""


class PredictionHead(nn.Module, ABC):
    @abstractmethod
    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Returns tensor appropriate for training (e.g. cube (B,C,H,W))."""


# --- Tokenizers --------------------------------------------------------------


@register_tokenizer("patch2d")
class Patch2DTokenizer(HyperspectralTokenizer):
    """
    ViT-style: split H×W into non-overlapping Ph×Pw patches; each patch carries full spectrum C.

    patch_h, patch_w must divide H, W.
    """

    def __init__(
        self,
        spatial_size: tuple[int, int],
        in_channels: int,
        embed_dim: int,
        patch_h: int,
        patch_w: int,
    ):
        super().__init__()
        h, w = spatial_size
        if h % patch_h != 0 or w % patch_w != 0:
            raise ValueError(f"spatial ({h},{w}) must be divisible by patch ({patch_h},{patch_w})")
        self.spatial_size = spatial_size
        self.in_channels = in_channels
        self.patch_h = patch_h
        self.patch_w = patch_w
        self.embed_dim = embed_dim
        self.num_tokens = (h // patch_h) * (w // patch_w)
        self.patch_volume = patch_h * patch_w * in_channels
        self.unfold = nn.Unfold(kernel_size=(patch_h, patch_w), stride=(patch_h, patch_w))
        self.proj = nn.Linear(self.patch_volume, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = self.unfold(x)
        u = u.transpose(1, 2)
        return self.proj(u)


@register_tokenizer("patch3d")
class Patch3DTokenizer(HyperspectralTokenizer):
    """
    Non-overlapping 3D patches over (H, W, C) with patch_c dividing C.

    Tokens ordered (grid_h, grid_w, grid_c).
    """

    def __init__(
        self,
        spatial_size: tuple[int, int],
        in_channels: int,
        embed_dim: int,
        patch_h: int,
        patch_w: int,
        patch_c: int,
    ):
        super().__init__()
        h, w = spatial_size
        c = in_channels
        if h % patch_h != 0 or w % patch_w != 0 or c % patch_c != 0:
            raise ValueError(
                f"shape ({h},{w},{c}) must be divisible by patch ({patch_h},{patch_w},{patch_c})"
            )
        self.spatial_size = spatial_size
        self.in_channels = in_channels
        self.patch_h = patch_h
        self.patch_w = patch_w
        self.patch_c = patch_c
        self.embed_dim = embed_dim
        self.grid_h = h // patch_h
        self.grid_w = w // patch_w
        self.grid_c = c // patch_c
        self.num_tokens = self.grid_h * self.grid_w * self.grid_c
        self.patch_volume = patch_h * patch_w * patch_c
        self.proj = nn.Linear(self.patch_volume, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Patch order: grid over (gh, gw, gc); each patch is (ph, pw, pc).
        b, _, _, _ = x.shape
        ph, pw, pc = self.patch_h, self.patch_w, self.patch_c
        gh, gw, gc = self.grid_h, self.grid_w, self.grid_c
        x = x.view(b, gc, pc, gh, ph, gw, pw)
        x = x.permute(0, 3, 5, 1, 4, 6, 2).contiguous()
        x = x.view(b, gh * gw * gc, ph * pw * pc)
        return self.proj(x)


@register_tokenizer("conv_stem")
class ConvStemTokenizer(HyperspectralTokenizer):
    """
    Single conv layer: Conv2d(C, D, kernel=stride=patch) acting as patch embed (hybrid ViT stem).
    """

    def __init__(
        self,
        spatial_size: tuple[int, int],
        in_channels: int,
        embed_dim: int,
        patch_size: int,
    ):
        super().__init__()
        h, w = spatial_size
        if h % patch_size != 0 or w % patch_size != 0:
            raise ValueError(f"spatial ({h},{w}) must divide patch_size {patch_size}")
        self.spatial_size = spatial_size
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_tokens = (h // patch_size) * (w // patch_size)
        self.patch_volume = patch_size * patch_size * in_channels
        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.proj(x)
        return z.flatten(2).transpose(1, 2)


# --- Positional encoding -----------------------------------------------------


@register_pos("learned")
class LearnedPositionalEncoding1D(PositionalEncoding):
    def __init__(self, num_tokens: int, embed_dim: int, use_cls_token: bool):
        super().__init__()
        n = num_tokens + (1 if use_cls_token else 0)
        self.pos = nn.Parameter(torch.zeros(1, n, embed_dim))
        nn.init.trunc_normal_(self.pos, std=0.02)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.shape[1] != self.pos.shape[1]:
            raise ValueError(
                f"Token length {tokens.shape[1]} != pos length {self.pos.shape[1]}"
            )
        return tokens + self.pos


@register_pos("sin_cos")
class SinCosPositionalEncoding1D(PositionalEncoding):
    """Fixed 1D sin-cos; matches sequence length."""

    def __init__(self, num_tokens: int, embed_dim: int, use_cls_token: bool):
        super().__init__()
        self.use_cls_token = use_cls_token
        n = num_tokens + (1 if use_cls_token else 0)
        pe = self._build_pe(n, embed_dim)
        self.register_buffer("pe", pe, persistent=False)

    @staticmethod
    def _build_pe(n: int, d: int) -> torch.Tensor:
        position = torch.arange(n, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d, 2, dtype=torch.float32) * (-torch.log(torch.tensor(10000.0)) / d))
        pe = torch.zeros(n, d)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])
        return pe.unsqueeze(0)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return tokens + self.pe.to(dtype=tokens.dtype, device=tokens.device)


# --- Backbone ----------------------------------------------------------------


class _PreLNBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm, need_weights=False)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


@register_backbone("preln_vit")
class PreLNTransformerEncoder(TransformerEncoderBackbone):
    """Stack of Pre-LN transformer blocks using scaled dot-product attention."""

    def __init__(
        self,
        embed_dim: int,
        depth: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        **_: Any,
    ):
        super().__init__()
        self.blocks = nn.ModuleList(
            [_PreLNBlock(embed_dim, num_heads, mlp_ratio, dropout) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x)


# --- Heads -------------------------------------------------------------------


@register_head("cube_fold_patch2d")
class CubeFoldPatch2DHead(PredictionHead):
    """
    Predict each patch's flattened spectral cube slice; fold back to (B,C,H,W).

    Requires tokenizer patch2d or conv_stem with square patch such that patch_volume matches.
    """

    def __init__(
        self,
        spatial_size: tuple[int, int],
        in_channels: int,
        embed_dim: int,
        patch_h: int,
        patch_w: int,
        use_cls_token: bool,
        **_: Any,
    ):
        super().__init__()
        self.h, self.w = spatial_size
        self.c = in_channels
        self.patch_h = patch_h
        self.patch_w = patch_w
        self.patch_volume = patch_h * patch_w * in_channels
        self.use_cls_token = use_cls_token
        self.proj = nn.Linear(embed_dim, self.patch_volume)
        self.fold = nn.Fold(
            output_size=spatial_size,
            kernel_size=(patch_h, patch_w),
            stride=(patch_h, patch_w),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if self.use_cls_token:
            tokens = tokens[:, 1:, :]
        # tokens: (B, N, D)
        b, n, _ = tokens.shape
        patch_flat = self.proj(tokens)
        patch_flat = patch_flat.transpose(1, 2)
        return self.fold(patch_flat)


@register_head("cube_fold_conv_stem")
class CubeFoldConvStemHead(PredictionHead):
    """Inverse of conv_stem spatial grid: ConvTranspose2d then outputs C channels."""

    def __init__(
        self,
        spatial_size: tuple[int, int],
        in_channels: int,
        embed_dim: int,
        patch_size: int,
        use_cls_token: bool,
        **_: Any,
    ):
        super().__init__()
        self.use_cls_token = use_cls_token
        self.patch_size = patch_size
        self.decoder = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
        )
        self.deconv = nn.ConvTranspose2d(
            embed_dim,
            in_channels,
            kernel_size=patch_size,
            stride=patch_size,
        )
        self.spatial_size = spatial_size

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if self.use_cls_token:
            tokens = tokens[:, 1:, :]
        b, n, d = tokens.shape
        h, w = self.spatial_size
        p = self.patch_size
        gh, gw = h // p, w // p
        x = self.decoder(tokens)
        x = x.transpose(1, 2).reshape(b, d, gh, gw)
        return self.deconv(x)


@register_head("cube_linear_patch3d")
class CubeLinearPatch3DHead(PredictionHead):
    """Inverse of Patch3DTokenizer fold: Linear then explicit reshape to cube."""

    def __init__(
        self,
        spatial_size: tuple[int, int],
        in_channels: int,
        embed_dim: int,
        patch_h: int,
        patch_w: int,
        patch_c: int,
        use_cls_token: bool,
        **_: Any,
    ):
        super().__init__()
        self.spatial_size = spatial_size
        self.in_channels = in_channels
        self.patch_h = patch_h
        self.patch_w = patch_w
        self.patch_c = patch_c
        self.use_cls_token = use_cls_token
        self.proj = nn.Linear(embed_dim, patch_h * patch_w * patch_c)

        h, w = spatial_size
        self.grid_h = h // patch_h
        self.grid_w = w // patch_w
        self.grid_c = in_channels // patch_c

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if self.use_cls_token:
            tokens = tokens[:, 1:, :]
        b = tokens.shape[0]
        gh, gw, gc = self.grid_h, self.grid_w, self.grid_c
        ph, pw, pc = self.patch_h, self.patch_w, self.patch_c
        patches = self.proj(tokens).view(b, gh * gw * gc, ph * pw * pc)
        patches = patches.view(b, gh, gw, gc, ph, pw, pc)
        patches = patches.permute(0, 3, 6, 1, 4, 2, 5).contiguous()
        return patches.view(b, gc * pc, gh * ph, gw * pw)


@register_head("identity")
class IdentityHead(PredictionHead):
    """Returns tokens unchanged (for contrastive / feature extraction only)."""

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return tokens


# --- Full model --------------------------------------------------------------


class HyperspectralViT(nn.Module):
    """
    tokenizer → optional CLS → positional encoding → backbone → head.

    Components are injected for swapping implementations without subclassing this class.
    """

    def __init__(
        self,
        cfg: ModelConfig,
        tokenizer: HyperspectralTokenizer,
        pos_encoding: PositionalEncoding,
        backbone: TransformerEncoderBackbone,
        head: Optional[PredictionHead],
    ):
        super().__init__()
        self.cfg = cfg
        self.tokenizer = tokenizer
        self.pos_encoding = pos_encoding
        self.backbone = backbone
        self.head = head
        self.use_cls_token = cfg.use_cls_token
        if cfg.use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, cfg.embed_dim))
            nn.init.trunc_normal_(self.cls_token, std=0.02)
        else:
            self.cls_token = None

    def forward(self, x: torch.Tensor) -> ModelOutput:
        """
        Args:
            x: (B, C, H, W) normalized cube.

        Returns:
            ModelOutput with encoded tokens and optional ``cube_reconstruction``.
        """
        tokens = self.tokenizer(x)
        if self.cls_token is not None:
            cls = self.cls_token.expand(x.shape[0], -1, -1)
            tokens = torch.cat([cls, tokens], dim=1)
        tokens = self.pos_encoding(tokens)
        encoded = self.backbone(tokens)
        cube_hat = None
        if self.head is not None and not isinstance(self.head, IdentityHead):
            cube_hat = self.head(encoded)
        elif isinstance(self.head, IdentityHead):
            pass
        return ModelOutput(tokens_encoded=encoded, cube_reconstruction=cube_hat, aux={})


def _validate_cfg(cfg: ModelConfig) -> None:
    h, w = cfg.spatial_size
    if h <= 0 or w <= 0 or cfg.in_channels <= 0:
        raise ValueError("Invalid spatial_size or in_channels")
    if cfg.embed_dim % cfg.num_heads != 0:
        raise ValueError("embed_dim must be divisible by num_heads for MultiheadAttention.")


def build_model(cfg: ModelConfig) -> HyperspectralViT:
    """Construct HyperspectralViT from config using registries."""
    _validate_cfg(cfg)
    if cfg.pos_encoding_type == "sin_cos" and cfg.embed_dim % 2 != 0:
        raise ValueError("sin_cos positional encoding requires embed_dim to be even.")

    tok_cls = TOKENIZER_REGISTRY.get(cfg.tokenizer_type)
    if tok_cls is None:
        raise KeyError(f"Unknown tokenizer_type {cfg.tokenizer_type!r}; have {list(TOKENIZER_REGISTRY)}")

    tok_kw = dict(cfg.tokenizer_kwargs)
    if cfg.tokenizer_type == "patch2d":
        tokenizer = tok_cls(
            cfg.spatial_size,
            cfg.in_channels,
            cfg.embed_dim,
            patch_h=tok_kw.pop("patch_h"),
            patch_w=tok_kw.pop("patch_w"),
            **tok_kw,
        )
    elif cfg.tokenizer_type == "patch3d":
        tokenizer = tok_cls(
            cfg.spatial_size,
            cfg.in_channels,
            cfg.embed_dim,
            patch_h=tok_kw.pop("patch_h"),
            patch_w=tok_kw.pop("patch_w"),
            patch_c=tok_kw.pop("patch_c"),
            **tok_kw,
        )
    elif cfg.tokenizer_type == "conv_stem":
        tokenizer = tok_cls(
            cfg.spatial_size,
            cfg.in_channels,
            cfg.embed_dim,
            patch_size=tok_kw.pop("patch_size"),
            **tok_kw,
        )
    else:
        raise ValueError(
            f"tokenizer_type {cfg.tokenizer_type!r} has no constructor mapping in build_model; "
            "register it and add a branch."
        )

    pos_cls = POS_REGISTRY.get(cfg.pos_encoding_type)
    if pos_cls is None:
        raise KeyError(f"Unknown pos_encoding_type {cfg.pos_encoding_type!r}")
    pos_encoding = pos_cls(
        tokenizer.num_tokens,
        cfg.embed_dim,
        cfg.use_cls_token,
        **cfg.pos_encoding_kwargs,
    )

    bb_cls = BACKBONE_REGISTRY.get(cfg.backbone_type)
    if bb_cls is None:
        raise KeyError(f"Unknown backbone_type {cfg.backbone_type!r}")
    backbone = bb_cls(
        embed_dim=cfg.embed_dim,
        depth=cfg.depth,
        num_heads=cfg.num_heads,
        mlp_ratio=cfg.mlp_ratio,
        dropout=cfg.dropout,
        **cfg.backbone_kwargs,
    )

    head: Optional[PredictionHead] = None
    if cfg.head_type and cfg.head_type.lower() != "none":
        hd_cls = HEAD_REGISTRY.get(cfg.head_type)
        if hd_cls is None:
            raise KeyError(f"Unknown head_type {cfg.head_type!r}")
        head_kw = dict(cfg.head_kwargs)
        if cfg.head_type == "cube_fold_patch2d":
            head = hd_cls(
                cfg.spatial_size,
                cfg.in_channels,
                cfg.embed_dim,
                patch_h=head_kw.pop("patch_h", cfg.tokenizer_kwargs.get("patch_h")),
                patch_w=head_kw.pop("patch_w", cfg.tokenizer_kwargs.get("patch_w")),
                use_cls_token=cfg.use_cls_token,
                **head_kw,
            )
        elif cfg.head_type == "cube_fold_conv_stem":
            head = hd_cls(
                cfg.spatial_size,
                cfg.in_channels,
                cfg.embed_dim,
                patch_size=head_kw.pop("patch_size", cfg.tokenizer_kwargs.get("patch_size")),
                use_cls_token=cfg.use_cls_token,
                **head_kw,
            )
        elif cfg.head_type == "cube_linear_patch3d":
            head = hd_cls(
                cfg.spatial_size,
                cfg.in_channels,
                cfg.embed_dim,
                patch_h=head_kw.pop("patch_h", cfg.tokenizer_kwargs.get("patch_h")),
                patch_w=head_kw.pop("patch_w", cfg.tokenizer_kwargs.get("patch_w")),
                patch_c=head_kw.pop("patch_c", cfg.tokenizer_kwargs.get("patch_c")),
                use_cls_token=cfg.use_cls_token,
                **head_kw,
            )
        elif cfg.head_type == "identity":
            head = hd_cls(**head_kw)
        else:
            head = hd_cls(**head_kw)

    return HyperspectralViT(cfg, tokenizer, pos_encoding, backbone, head)


def describe_components(model: HyperspectralViT) -> list[ComponentSpec]:
    """Summarize tensor shapes for logging."""
    cfg = model.cfg
    h, w = cfg.spatial_size
    c = cfg.in_channels
    tok = model.tokenizer
    in_spec = TensorSpec("NCHW", ("B", c, h, w), "input cube")
    out_spec = TensorSpec("BNC", ("B", tok.num_tokens + cfg.use_cls_token, cfg.embed_dim), "encoded sequence")
    return [
        ComponentSpec(
            name="tokenizer",
            in_spec=in_spec,
            out_spec=TensorSpec("BNC", ("B", tok.num_tokens, cfg.embed_dim), "patch tokens"),
            embed_dim=cfg.embed_dim,
            num_tokens=tok.num_tokens,
        ),
        ComponentSpec(
            name="backbone",
            in_spec=out_spec,
            out_spec=out_spec,
            embed_dim=cfg.embed_dim,
            num_tokens=tok.num_tokens + cfg.use_cls_token,
        ),
    ]
