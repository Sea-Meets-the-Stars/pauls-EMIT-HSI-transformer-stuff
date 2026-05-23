### Purpose: Define a simple ViT encoder for hyperspectral images, pretrained using MAE masking

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import h5py
import math


def sinusoidal_encoding_1d(positions, dim):
    # positions: (L,) float — x_g, y_g, or λ_scaled
    # returns: (L, dim)
    max_period = 10000.0
    half = dim // 2
    
    freq_indices = torch.arange(half, device=positions.device, dtype=torch.float32)
    div_term = torch.exp(freq_indices * 2.0 * (-math.log(max_period) / dim))
    
    angle = positions.float().unsqueeze(1) * div_term.unsqueeze(0)
    pe = torch.zeros(positions.shape[0], dim, device=positions.device)
    
    pe[:, 0::2] = torch.sin(angle)
    pe[:, 1::2] = torch.cos(angle)
    
    return pe

class SimpleHyperspectralMAEEncoder(nn.Module):
    def __init__(self, embed_dim = 512, patch_size_spatial = 16, patch_size_spectral = 15, num_heads = 8, 
                 num_encoder_blocks = 4, num_decoder_blocks = 2, mlp_ratio = 4.0, masking_ratio = 0.75):
        super().__init__()
        # Note: Assuming Hypercubes are (Batch, Channels, Height, Width)
        
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
        
        ## ======================== ##
        ## Patch Embedding options: ##
        ## ======================== ##
        #TODO: Verify if dimensionality works for these:
        self.patch_embed_3d_conv = nn.Conv3d(1, self.encoder_embed_dim, 
                                             kernel_size=(self.patch_size_spectral, self.patch_size_spatial, self.patch_size_spatial), 
                                             stride=(self.patch_size_spectral, self.patch_size_spatial, self.patch_size_spatial))
        
        self.patch_embed_3d_linear = nn.Linear(self.patch_size_spectral * self.patch_size_spatial * self.patch_size_spatial * self.num_bands, 
                                               self.encoder_embed_dim)
        
        
        ## ============================ ##
        ## Positional Encoding options: ##
        ## ============================ ##
        #TODO: Verify if dimensionality works for these:
        # Use sinusoidal positional encoding:
        
        # Add: factorized tables forward can index or broadcast.
        # Spatial: patch grid coordinates (not patch_size)
        x_idx = torch.arange(self.num_patches_w, dtype=torch.float32)   # 0..7
        y_idx = torch.arange(self.num_patches_h, dtype=torch.float32)   # 0..7
        
        # Spectral: index scaling replaces λ (equally spaced → same as normalized c)
        # Matches Eq. (7) with c/(C'-1) in [0,1] then × N_spatial
        denom = max(self.num_patches_c - 1, 1)
        c_scaled = torch.arange(self.num_patches_c, dtype=torch.float32) * (self.n_spatial / denom)
        
        self.register_buffer("pe_x", sinusoidal_encoding_1d(x_idx, self.encoder_embed_dim))       # (Wp, D)
        self.register_buffer("pe_y", sinusoidal_encoding_1d(y_idx, self.encoder_embed_dim))       # (Hp, D)
        self.register_buffer("pe_spectral", sinusoidal_encoding_1d(c_scaled, self.encoder_embed_dim))  # (Cp, D)
        
        
        #TODO: Audit to make sure this is right.  This is from an LLM raw:
        g_idx = torch.arange(self.num_patches_h * self.num_patches_w, dtype=torch.float32)
        x_g = (g_idx % self.num_patches_w).long()
        y_g = (g_idx // self.num_patches_w).long()
        e_spatial = sinusoidal_encoding_1d(x_g.float(), self.encoder_embed_dim) + sinusoidal_encoding_1d(
            y_g.float(), self.encoder_embed_dim
        )  # (G, D) with G = 64
        e_pos = self.pe_spectral[:, None, :] + e_spatial[None, :, :]  # (C′, G, D)
        self.register_buffer(
            "pos_embed",
            e_pos.reshape(self.num_tokens, self.encoder_embed_dim),
        )  # (1216, D)
        
        
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
        self.patch_volume = (
            self.patch_size_spectral
            * self.patch_size_spatial
            * self.patch_size_spatial
        )  # 15 * 16 * 16 = 3840
        self.pred_head = nn.Linear(self.encoder_embed_dim, self.patch_volume)
        