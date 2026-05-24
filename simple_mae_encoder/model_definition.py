### Purpose: Define a simple ViT encoder for hyperspectral images, pretrained using MAE masking
# Note:  Dimensions are (B, H, W, C)

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import h5py
import math


def sinusoidal_encoding_1d(positions, dim):
    half = dim // 2
    freq = torch.exp(
        torch.arange(half, device=positions.device, dtype=torch.float32)
        * (-math.log(10000.0) / dim)
    )
    angles = positions.float().unsqueeze(1) * freq.unsqueeze(0)
    pe = torch.zeros(positions.shape[0], dim, device=positions.device)
    pe[:, 0::2] = torch.sin(angles)
    pe[:, 1::2] = torch.cos(angles)
    return pe

def build_3d_pos_embed(num_h, num_w, num_c, dim):
    """
    Create 3D positional encoding for input order: Batch, Height, Width, Channels.
    The returned embeddings are ordered as (h, w, c) to match patchify-flatten order with this layout.
    """
    h = torch.arange(num_h, dtype=torch.float32)
    w = torch.arange(num_w, dtype=torch.float32)
    c = torch.arange(num_c, dtype=torch.float32)
    pe_h = sinusoidal_encoding_1d(h, dim)  # (H, D)
    pe_w = sinusoidal_encoding_1d(w, dim)  # (W, D)
    pe_c = sinusoidal_encoding_1d(c, dim)  # (C, D)
    # Token order (h, w, c) matches patchify flatten order for BHWC
    pos = (
        pe_h[:, None, None, :]     # (H, 1, 1, D)
        + pe_w[None, :, None, :]   # (1, W, 1, D)
        + pe_c[None, None, :, :]   # (1, 1, C, D)
    )  # (H, W, C, D)
    return pos.reshape(num_h * num_w * num_c, dim)


class SimpleHyperspectralMAEEncoder(nn.Module):
    def __init__(self, embed_dim = 512, patch_size_spatial = 16, patch_size_spectral = 15, num_heads = 8, 
                 num_encoder_blocks = 4, num_decoder_blocks = 2, mlp_ratio = 4.0, masking_ratio = 0.75):
        super().__init__()
        # Input layout: (Batch, Height, Width, Channels)
        
        self.patch_size_spatial = patch_size_spatial
        self.patch_size_spectral = patch_size_spectral
        self.encoder_embed_dim = embed_dim
        self.decoder_embed_dim = embed_dim   #in case we want to do a paper style thing with different dims
        self.num_heads = num_heads
        self.num_encoder_blocks = num_encoder_blocks
        self.num_decoder_blocks = num_decoder_blocks
        self.mlp_ratio = mlp_ratio
        self.masking_ratio = masking_ratio
        
        # Constants:
        self.chip_size_spatial = 128
        self.chip_size_spectral = 285
        self.num_bands = self.chip_size_spectral
        
        assert self.encoder_embed_dim % 2 == 0
        assert self.decoder_embed_dim % 2 == 0
        assert self.chip_size_spatial % self.patch_size_spatial == 0
        assert self.chip_size_spectral % self.patch_size_spectral == 0
        assert self.num_bands % self.patch_size_spectral == 0
        
        # Derived constants:
        self.num_patches_h = self.chip_size_spatial // self.patch_size_spatial   # 8
        self.num_patches_w = self.chip_size_spatial // self.patch_size_spatial   # 8
        self.num_patches_c = self.num_bands // self.patch_size_spectral  # 19
        self.num_tokens = self.num_patches_c * self.num_patches_h * self.num_patches_w  # 1216
        self.n_spatial = max(self.num_patches_h, self.num_patches_w) # 8
        self.patch_volume = (
            self.patch_size_spectral
            * self.patch_size_spatial
            * self.patch_size_spatial
        )  # 15 * 16 * 16 = 3840
        
        ## ======================== ##
        ## Patch Embedding options: ##
        ## ======================== ##
        #TODO: Verify if dimensionality works for these:
        self.patch_embed_3d_conv = nn.Conv3d(1, self.encoder_embed_dim, 
                                             kernel_size=(self.patch_size_spectral, self.patch_size_spatial, self.patch_size_spatial), 
                                             stride=(self.patch_size_spectral, self.patch_size_spatial, self.patch_size_spatial))
        
        self.patch_embed_3d_linear = nn.Linear(self.patch_volume, self.encoder_embed_dim)
        
        
        ## ============================ ##
        ## Positional Encoding options: ##
        ## ============================ ##
        #TODO: Verify if dimensionality works for these:
        # Use sinusoidal positional encoding:
        
        pos_embed = build_3d_pos_embed(
            self.num_patches_h, self.num_patches_w, self.num_patches_c, self.encoder_embed_dim
        )
        self.register_buffer("pos_embed", pos_embed)
        
        
        ## ============================ ##
        ## Transformer Encoder options: ##
        ## ============================ ##
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.encoder_embed_dim,
            nhead=self.num_heads,
            dim_feedforward=int(self.encoder_embed_dim * self.mlp_ratio),
            batch_first=True,       # (B, N, D) in forward
            activation="gelu",      # explicit (ViT/MAE default)
            norm_first=False,       # explicit pre-LN vs post-LN choice
        )
        
        self.encoder = nn.TransformerEncoder(
            self.encoder_layer,
            num_layers=self.num_encoder_blocks
        )
        
        ## ============================ ##
        ## Transformer Decoder options: ##
        ## ============================ ##
        self.decoder_layer = nn.TransformerEncoderLayer(
            d_model=self.decoder_embed_dim,
            nhead=self.num_heads,
            dim_feedforward=int(self.decoder_embed_dim * self.mlp_ratio),
            batch_first=True,       # (B, N, D) in forward
            activation="gelu",      # explicit (ViT/MAE default)
            norm_first=False,       # explicit pre-LN vs post-LN choice
        )
        
        self.decoder = nn.TransformerEncoder(
            self.decoder_layer,
            num_layers=self.num_decoder_blocks
        )
        
        ## ============================ ##
        ## Masked Autoencoder options: ##
        ## ============================ ##
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.encoder_embed_dim))
        nn.init.normal_(self.mask_token, std=0.02)
        
        # Reconstruct voxel values in each (s × p × p) patch
        
        self.pred_head = nn.Linear(self.encoder_embed_dim, self.patch_volume)

    def patchify(self, x):
        # (B, H, W, C) -> tokens in (h, w, c) order
        batch_size = x.shape[0]
        x = x.reshape(
            batch_size,
            self.num_patches_h,
            self.patch_size_spatial,
            self.num_patches_w,
            self.patch_size_spatial,
            self.num_patches_c,
            self.patch_size_spectral,
        )
        x = x.permute(0, 1, 3, 5, 2, 4, 6)  # (B, Hp, Wp, Cp, ps_h, ps_w, ps_c)
        return x.reshape(batch_size, self.num_tokens, self.patch_volume)

    def unpatchify(self, patches):
        batch_size = patches.shape[0]
        x = patches.reshape(
            batch_size,
            self.num_patches_h,
            self.num_patches_w,
            self.num_patches_c,
            self.patch_size_spatial,
            self.patch_size_spatial,
            self.patch_size_spectral,
        )
        x = x.permute(0, 1, 4, 2, 5, 3, 6)  # (B, Hp, ps_h, Wp, ps_w, Cp, ps_c)
        return x.reshape(
            batch_size,
            self.chip_size_spatial,
            self.chip_size_spatial,
            self.num_bands,
        )  # (B, H, W, C)

    def forward(self, x):
        batch_size = x.shape[0]

        ## ======================== ##
        ## Patchify + linear embed ##
        ## ======================== ##
        patches = self.patchify(x)
        tokens = self.patch_embed_3d_linear(patches)
        tokens = tokens + self.pos_embed.unsqueeze(0)

        ## ============================ ##
        ## Spatial-spectral masking    ##
        ## ============================ ##
        shuffle = torch.rand(batch_size, self.num_tokens, device=tokens.device).argsort(dim=1)
        len_visible = int(self.num_tokens * (1.0 - self.masking_ratio))
        ids_visible = shuffle[:, :len_visible]

        encoder_in = torch.gather(
            tokens, 1, ids_visible.unsqueeze(-1).expand(-1, -1, self.encoder_embed_dim)
        )
        encoder_out = self.encoder(encoder_in)

        ## ============================ ##
        ## Decoder + reconstruction    ##
        ## ============================ ##
        decoder_in = self.mask_token.expand(batch_size, self.num_tokens, self.encoder_embed_dim).clone()
        decoder_in.scatter_(
            1, ids_visible.unsqueeze(-1).expand(-1, -1, self.encoder_embed_dim), encoder_out
        )
        decoder_in = decoder_in + self.pos_embed.unsqueeze(0)
        decoder_out = self.decoder(decoder_in)
        pred_patches = self.pred_head(decoder_out)

        ## ============================ ##
        ## Holistic loss (all tokens)  ##
        ## ============================ ##
        loss = F.mse_loss(pred_patches, patches)
        return loss

    @torch.no_grad()
    def reconstruct(self, x):
        batch_size = x.shape[0]
        patches = self.patchify(x)
        tokens = self.patch_embed_3d_linear(patches) + self.pos_embed.unsqueeze(0)

        shuffle = torch.rand(batch_size, self.num_tokens, device=tokens.device).argsort(dim=1)
        len_visible = int(self.num_tokens * (1.0 - self.masking_ratio))
        ids_visible = shuffle[:, :len_visible]

        encoder_in = torch.gather(
            tokens, 1, ids_visible.unsqueeze(-1).expand(-1, -1, self.encoder_embed_dim)
        )
        encoder_out = self.encoder(encoder_in)

        decoder_in = self.mask_token.expand(batch_size, self.num_tokens, self.encoder_embed_dim).clone()
        decoder_in.scatter_(
            1, ids_visible.unsqueeze(-1).expand(-1, -1, self.encoder_embed_dim), encoder_out
        )
        decoder_in = decoder_in + self.pos_embed.unsqueeze(0)
        pred_patches = self.pred_head(self.decoder(decoder_in))

        return self.unpatchify(pred_patches), self.unpatchify(patches)