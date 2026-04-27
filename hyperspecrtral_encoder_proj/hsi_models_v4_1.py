import os, sys
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # only expose one GPU to PyTorch
import numpy as np
import glob
import random
import gc

import math
from datetime import datetime
import multiprocessing as mp

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from einops import rearrange

# from transformers import AutoConfig, AutoProcessor, Qwen3VLForConditionalGeneration
# qwen_name = "Qwen/Qwen3-VL-2B-Instruct"
# hidden_size = AutoConfig.from_pretrained(qwen_name).text_config.hidden_size

BAND_STATS_PATH = "emit_band_stats.npz" 
INSTRUMENT_PATH = "instrument.json"
FILL = -9999.0
BANDS = 285

### Model Classes:

class SimpleTransformerBlock(nn.Module):
    """
    Repeatable transformer block.

    Shapes:
      - token_embeddings: (batch_size, num_tokens, embed_dim)
    """
    def __init__(self, embed_dim: int, num_heads: int = 8, mlp_ratio: float = 4.0, dropout: float = 0.0, do_initial_norm: bool = True):
        """
        embed_dim: the token embedding size D. Every token is a length-D vector. 
                   this is intended to equal Qwens text hidden size (e.g., 2048).
        num_heads: number of attention heads. Multi-head attention splits the D-dim vector into num_heads subspaces 
                   (each head has size D / num_heads).
        mlp_ratio: how wide the MLP “expansion” is relative to D. 
                   ex: if mlp_ratio=4.0, the MLP hidden layer is size 4*D.
        dropout:   dropout probability used inside attention
        do_initial_norm: whether to apply initial normalization to the token embeddings
        """
        super().__init__()
        if do_initial_norm:
            self.pre_attn_norm = nn.LayerNorm(embed_dim)
        else:
            self.pre_attn_norm = nn.Identity()
        #self.pre_attn_norm = nn.LayerNorm(embed_dim)   #Drop this to preserve brightness information # Pre-normalize each token's features to stabilize training.
                                                        # For each token vector of length D, it normalizes across the feature dimension (not across tokens), 
                                                        # producing roughly zero-mean/unit-variance features with learnable scale/shift
        self.self_attn = nn.MultiheadAttention(         # Self-attention layer, computes a weighted mixture of all tokens in the sequence
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,       #batch_first=True means tensors are shaped (B, N, D) instead of (N, B, D).
        )
        self.pre_mlp_norm = nn.LayerNorm(embed_dim)     # Re-normalize before MLP
        mlp_hidden_dim = int(embed_dim * mlp_ratio)     # MLP hidden layer size
        self.mlp = nn.Sequential(                       # Feed-Forward block
            nn.Linear(embed_dim, mlp_hidden_dim),           # Expand hidden dim
            nn.GELU(),                                      # Nonlinearity
            nn.Linear(mlp_hidden_dim, embed_dim),           # Project back to hidden dim
        )

    def forward(self, token_embeddings: torch.Tensor) -> torch.Tensor:
        # token_embeddings: (batch_size, num_tokens, embed_dim)
        attn_input = self.pre_attn_norm(token_embeddings)
        #attn_input = token_embeddings
        attn_output, _ = self.self_attn(attn_input, attn_input, attn_input, need_weights=False) #query = key = value = attn_input means “each token attends to all tokens in the same sequence.”
                                                                                                # Output attn_output is a new token sequence where each token is a learned mixture of other tokens.
                                                                                                # _ would be attention weights, but need_weights=False skips returning them (saves compute/memory).
        token_embeddings = token_embeddings + attn_output   #Residual connection: the block learns a delta to add to the input rather than replacing it entirely.

        mlp_input = self.pre_mlp_norm(token_embeddings)
        token_embeddings = token_embeddings + self.mlp(mlp_input)   #Applies the token-wise MLP (same MLP to every token independently) and adds it back via another residual.

        return token_embeddings


class HyperspectralVisionEncoder(nn.Module):
    """
    Hyperspectral "vision tower" whose output is designed to be *injected into Qwen*.

    Integration contract (see `DualVisionQwen`):
      forward(pixel_values, grid_thw=...) -> (flattened_token_embeddings, deepstack)

    - pixel_values: (batch_size, num_bands, image_h, image_w)   (here num_bands=285)
    - flattened_token_embeddings: (total_tokens_across_batch, embed_dim)
        where total_tokens_across_batch = batch_size * tokens_per_image_after_merge
    - deepstack: list of optional intermediate features (unused here => [])

    Baseline assumptions:
      - Fixed 256x256 inputs when patch_size=16 (because pos_embed is sized for 16x16 tokens).
      - patch_size divides image_h and image_w exactly.
      - `grid_thw` is accepted for Qwen-compatibility but not used for computation/validation.
    """
    def __init__(
        self,
        embed_dim: int,
        num_bands: int = 285,
        num_spectral_queries: int = 4,
        patch_size_spatial: int = 16,
        patch_size_spectral: int = 15,
        depth: int = 6,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        do_query_summarization: bool = True,
        overlap_patches_spatial: bool = False,
        overlap_patches_spectral: bool = False,
    ):
        """
        embed_dim:  token embedding size D. This should match Qwens text hidden size (e.g., 2048), because these tokens will be injected into Qwens text stream.
        num_bands:  input channel count C for hyperspectral imagery (For EMIT: 285).
        patch_size: patch side length P in pixels. The model turns a H×W image into a (H/P)×(W/P) grid of tokens.
        spatial_merge_size: merge factor m. After encoding, it averages over each m×m block of tokens to reduce token count (like “spatial pooling” on tokens).
        depth:      number of Transformer blocks stacked.
        num_heads:  attention heads per Transformer block.
        """
        super().__init__()
        self.hidden_size = embed_dim
        self.patch_size_spatial = patch_size_spatial
        self.patch_size_spectral = patch_size_spectral
        self.do_query_summarization = do_query_summarization
        self.overlap_patches_spatial = overlap_patches_spatial
        self.overlap_patches_spectral = overlap_patches_spectral
        
        if self.do_query_summarization:
            self.num_spectral_queries = num_spectral_queries
        else:
            if self.overlap_patches_spectral:
                # Use same formula as Encoder to calculate raw segments
                spectral_stride = patch_size_spectral // 2
                self.num_spectral_queries = (num_bands - patch_size_spectral) // spectral_stride + 1
            else:
                self.num_spectral_queries = num_bands // patch_size_spectral
            num_spectral_queries = self.num_spectral_queries
        
        #self.spatial_merge_size = spatial_merge_size
        
        if self.overlap_patches_spatial:
            # Stride is half the patch size (e.g., 16 -> 8)
            self.spatial_stride = patch_size_spatial // 2
            # We add padding to ensure the grid comes out as a power of 2 (32x32)
            # Formula: (256 + 2*pad - kernel) / stride + 1 = 32
            self.spatial_padding = patch_size_spatial // 4 
        else:
            self.spatial_stride = patch_size_spatial
            self.spatial_padding = 0
            
        if self.overlap_patches_spectral:
            self.spectral_stride = patch_size_spectral // 2
        else:
            self.spectral_stride = patch_size_spectral
            
            
        # Calculate expected grid size (e.g., 16 or 32)
        # Assuming 256x256 image input
        self.grid_size = (256 + 2 * self.spatial_padding - self.patch_size_spatial) // self.spatial_stride + 1


        # 1. 3D Patch Embedding
        # Input: (B, 1, 285, 256, 256) -> Output: (B, D, 19, 16, 16)
        assert num_bands % patch_size_spectral == 0, "num_bands must be divisible by patch_size_spectral"
        self.patch_embed = nn.Conv3d(
            in_channels=1,
            out_channels=embed_dim,
            kernel_size=(patch_size_spectral, patch_size_spatial, patch_size_spatial),
            stride=(self.spectral_stride, self.spatial_stride, self.spatial_stride),
            padding=(0, self.spatial_padding, self.spatial_padding)
        )
        
        # 2. Spectral Bottleneck (Local Attention)
        if self.do_query_summarization:
            # Learnable Positional Embedding for the spectral bottleneck attention
            self.raw_segment_pos_embed = nn.Parameter(torch.randn(1, 1, num_bands // patch_size_spectral, embed_dim))
            nn.init.normal_(self.raw_segment_pos_embed, std=0.02)
            
            # Learnable Queries to summarize the 19 raw spectral segments into 4.
            self.spectral_queries = nn.Parameter(torch.randn(1, 1, num_spectral_queries, embed_dim))
            nn.init.normal_(self.spectral_queries, std=0.02)
            
            self.bottleneck_attn = nn.MultiheadAttention(embed_dim, num_heads=8, batch_first=True)
            self.bottleneck_norm = nn.LayerNorm(embed_dim)
        else:
            self.raw_segment_pos_embed = None
            self.spectral_queries = None
            self.bottleneck_attn = None
            self.bottleneck_norm = None
        
        # 3. Spatial and Spectral Positional Embeddings
        # Spatial: 16x16 = 256 locations
        self.spatial_pos_embed = nn.Parameter(torch.randn(1, self.grid_size * self.grid_size, 1, embed_dim))
        nn.init.normal_(self.spatial_pos_embed, std=0.02)
        
        # Spectral: num_spectral_queries locations
        self.spectral_pos_embed = nn.Parameter(torch.randn(1, 1, num_spectral_queries, embed_dim))
        nn.init.normal_(self.spectral_pos_embed, std=0.02)
        
        # Class Token:
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        nn.init.normal_(self.cls_token, std=0.02) 

        self.transformer_blocks = nn.ModuleList(                                    # Initializes Transformer Block Stack
            [
                SimpleTransformerBlock(embed_dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, do_initial_norm=False)
            ] + [
                SimpleTransformerBlock(embed_dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, do_initial_norm=True) for _ in range(depth-1)
            ]
        )
        self.final_norm = nn.LayerNorm(embed_dim)                                    # Final normalization layer.



    def forward(self, pixel_values: torch.Tensor, grid_thw=None, **kwargs):
        """
        This function takes a hyperspectral image (pixel_values) and encodes it into a sequence of tokens.
        It does this by:
        1. Patchifying the image into a grid of patches.
        2. Projecting each patch into a token embedding space.
        3. Adding positional embeddings to the tokens.
        4. Passing the tokens through a stack of transformer blocks.
        5. Merging the spatial tokens.
        6. Flattening the tokens for masked scatter injection.
        
        Inputs:
            pixel_values: hyperspectral batch tensor of shape (batch_size, num_bands, image_h, image_w)
                          For EMIT chips of size 256x256: (B, 285, 256, 256)
            grid_thw: (unused)metadata Qwen uses elsewhere, shaped (num_images, 3) with rows [t, h_grid, w_grid] in patch-grid units.
                      In this encoder, it’s accepted for compatibility but not used yet.
        Outputs:
            flattened_token_embeddings: (total_tokens_across_batch, embed_dim)
            deepstack: list of optional intermediate features (not implemented yet)
            
        Variable Shorthand:
            B = batch_size
            H = image_h
            W = image_w
            C = num_bands
            D = embed_dim (should match Qwens text hidden size, e.g., 2048)
            Hg = patch_grid_h
            Wg = patch_grid_w
        """
        
        batch_size = pixel_values.shape[0]
        
        # Add channel dim for Conv3d: (B, 285, 256, 256) -> (B, 1, 285, 256, 256)
        pixel_values = pixel_values.unsqueeze(1)
        
        # 1. Patchify -> (B, D, 19, 16, 16)
        patch_features = self.patch_embed(pixel_values)
        
        # Rearrange to isolate the "Spectral Sequence" per patch for attention
        # (Batch * Spatial_Grid, Raw_Spectral_Segments, Dim)
        patch_features = rearrange(patch_features, 'b d s h w -> (b h w) s d') #(B, D, 19, 16, 16) -> (B * 16 * 16, 19, D)
        
        # 2. Apply Spectral Bottleneck
        if self.do_query_summarization:
            # Broadcast (1, 19, D) to (Batch*Spatial, 19, D)
            patch_features = patch_features + self.raw_segment_pos_embed.squeeze(0)
            
            # Queries: Expand (1, 1, num_spectral_queries, D) -> (B*256, num_spectral_queries, D)
            num_spatial = patch_features.shape[0] // batch_size
            spectral_queries = self.spectral_queries.expand(batch_size, num_spatial, -1, -1)
            
            # Flatten to match keys: (Batch * Spatial, Q, D)
            spectral_queries = spectral_queries.flatten(0, 1)
            
            # Keys/Values: The 19 raw spectral segments
            spectral_summary, _ = self.bottleneck_attn(query=spectral_queries, key=patch_features, value=patch_features)
            spectral_summary = self.bottleneck_norm(spectral_summary) # (B*256, 4, D)
        else:
            spectral_summary = patch_features
        
        # 3. Restore Grid Structure
        # (B*256, 4, D) -> (B, 256, 4, D)
        x = rearrange(spectral_summary, '(b n) k d -> b n k d', b=batch_size)
        
        # 4. Add Factorized Positional Embeddings
        x = x + self.spatial_pos_embed   # Broadcasts over spectral dim
        x = x + self.spectral_pos_embed  # Broadcasts over spatial dim
        
        # 5. Flatten for Spatial Transformer -> (B, 1024, D)
        x = x.flatten(1, 2) 
        
        # Expand (1, 1, D) -> (Batch, 1, D)
        cls_token = self.cls_token.expand(batch_size, -1, -1)
        # Concatenate to front: (B, 1 + Total_Tokens, D)
        x = torch.cat((cls_token, x), dim=1)
        
        # 6. Global Processing
        for block in self.transformer_blocks:
            x = block(x)
        #x = self.final_norm(x)
        
        return x


class HyperspectralAutoencoder(nn.Module):
    """
    Reconstruction pretraining wrapper around `HyperspectralVisionEncoder`.

    HS cube -> HS tokens -> lightweight "unmerge + deconv" decoder -> reconstructed HS cube.

    Note: spatial mean-merge is lossy; repeating tokens is only an approximate inverse.
    """
    def __init__(
        self,
        embed_dim: int,
        num_bands: int = 285,
        num_spectral_queries: int = 4,
        patch_size_spatial: int = 16,
        patch_size_spectral: int = 15,
        depth: int = 6,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        decoder_input_channels: int = 2048,
        do_query_summarization: bool = True,
        overlap_patches_spatial: bool = False,
        overlap_patches_spectral: bool = False,
    ):
        """
        embed_dim:  token embedding size D. This should match Qwens text hidden size (e.g., 2048)
        num_bands:  input channel count C for hyperspectral imagery (For EMIT: 285).
        patch_size: patch side length P in pixels. The model turns a H×W image into a (H/P)×(W/P) grid of tokens.
        spatial_merge_size: merge factor m. After encoding, it averages over each m×m block of tokens to reduce token count (like “spatial pooling” on tokens).
        depth:      number of Transformer blocks stacked in the encoder.
        num_heads:  attention heads per Transformer block.
        """
        
        super().__init__()
        self.hsi_encoder = HyperspectralVisionEncoder(
            embed_dim=embed_dim,
            num_bands=num_bands,
            num_spectral_queries=num_spectral_queries,
            patch_size_spatial=patch_size_spatial,
            patch_size_spectral=patch_size_spectral,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            do_query_summarization=do_query_summarization,
            overlap_patches_spatial=overlap_patches_spatial,
            overlap_patches_spectral=overlap_patches_spectral
        )
        self.hidden_size = embed_dim
        self.patch_size_spatial = patch_size_spatial
        self.patch_size_spectral = patch_size_spectral
        self.do_query_summarization = do_query_summarization
        self.overlap_patches_spatial = overlap_patches_spatial
        self.overlap_patches_spectral = overlap_patches_spectral
        
        if self.do_query_summarization:
            self.num_spectral_queries = num_spectral_queries
        else:
            if self.overlap_patches_spectral:
                # Use same formula as Encoder
                spectral_stride = patch_size_spectral // 2
                self.num_spectral_queries = (num_bands - patch_size_spectral) // spectral_stride + 1
            else:
                self.num_spectral_queries = num_bands // patch_size_spectral
            num_spectral_queries = self.num_spectral_queries
            
        
        
        # 2. Decoder Input Projection
        # The encoder outputs 1024 tokens: (256 spatial * 4 spectral).
        # We need to map the 4 spectral tokens per pixel back into a feature map.
        # We map (4 * embed_dim) -> decoder_dim (256)
        self.dec_input_channels = decoder_input_channels # Initial channels for decoder
        
        self.decoder_projector = nn.Sequential(
            nn.Linear(self.hidden_size * self.num_spectral_queries, self.dec_input_channels),
            nn.LayerNorm(self.dec_input_channels),
            nn.GELU()
        )
        
        # New "Resize-Conv" Block (Replaces PixelShuffle)
        def upsample_block(in_ch: int, out_ch: int):
            return nn.Sequential(
                # 1. Upsample (Nearest is sharper, Bilinear is smoother)
                # We use Nearest to preserve "hard" distinct features initially, 
                # or Bilinear to avoid blockiness. Let's start with Bilinear for smoothness.
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                
                # 2. Refinement (The "Paint" brush)
                # A standard 3x3 convolution to fill in details on the larger grid.
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
                nn.GroupNorm(8, out_ch),
                nn.GELU(),
                
                # 3. Second Refinement (Optional, adds sharpness)
                nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
                nn.GELU(),
            )
            
            
        # ---------------------------------------------------------
        # Dynamic Decoder Schedule
        # ---------------------------------------------------------
        # 1. Calculate Stride and Initial Resolution
        stride = self.patch_size_spatial // 2 if self.overlap_patches_spatial else self.patch_size_spatial
        # Grid size matches the encoder's output grid (e.g., 256 // 16 = 16)
        initial_resolution = 256 // stride
        
        # 2. Calculate how many 2x upsamples we need to reach 256
        # e.g., if start is 16, log2(256/16) = 4 steps
        # e.g., if start is 32, log2(256/32) = 3 steps
        num_upsamples = int(math.log2(256 // initial_resolution))
        
        # 3. Construct the list of blocks dynamically
        layers = []
        current_ch = self.dec_input_channels
        min_ch = 256 # Floor for channel width so we don't get too thin
        
        for i in range(num_upsamples):
            # Target channels for next layer: Halve current, but respect floor
            next_ch = max(min_ch, current_ch // 2)
            
            # If this is the LAST block, ensure we match the expected final width 
            # (Optional: enforces strict alignment with your old logic, though generally not strictly required)
            if i == num_upsamples - 1:
                next_ch = min_ch
                
            layers.append(upsample_block(current_ch, next_ch))
            current_ch = next_ch

        self.decoder_ps = nn.Sequential(*layers)
        
        # Final projection to bands
        final_ch = max(256, decoder_input_channels // 8)

        # ---------------------------------------------------------
        # Final Output Projection
        # ---------------------------------------------------------
        # Use current_ch (the output of the last upsample block) as input here.
        # This ensures dimensions always match, no matter what loop logic happened above.
        self.decoder_out = nn.Sequential(
            nn.Conv2d(final_ch, final_ch, kernel_size=1), 
            nn.GELU(),
            nn.Conv2d(final_ch, num_bands, kernel_size=1) 
        )
        
        # Classifier Head:
        # Maps the single vector (embed_dim) to a single logit (plume probability)
        self.plume_classifier = nn.Linear(embed_dim, 1)
            
        # # ---------------------------------------------------------
        # # Decoder Schedule (Variable depth based on patch size)
        # # ---------------------------------------------------------
        # if self.patch_size_spatial == 16:
        #     # Grid starts at 16x16. Needs 4 upsamples to reach 256.
        #     self.decoder_ps = nn.Sequential(
        #         upsample_block(self.dec_input_channels, max(1024, decoder_input_channels // 2)),   # 16 -> 32
        #         upsample_block(max(1024, decoder_input_channels // 2), max(512, decoder_input_channels // 4)), # 32 -> 64
        #         upsample_block(max(512, decoder_input_channels // 4), max(256, decoder_input_channels // 8)), # 64 -> 128
        #         upsample_block(max(256, decoder_input_channels // 8), max(256, decoder_input_channels // 8)), # 128 -> 256
        #     )
        # elif self.patch_size_spatial == 8:
        #     # Grid starts at 32x32. Needs 3 upsamples to reach 256.
        #     self.decoder_ps = nn.Sequential(
        #         upsample_block(self.dec_input_channels, max(1024, decoder_input_channels // 2)),   # 32 -> 64
        #         upsample_block(max(1024, decoder_input_channels // 2), max(512, decoder_input_channels // 4)), # 64 -> 128
        #         upsample_block(max(512, decoder_input_channels // 4), max(256, decoder_input_channels // 8)), # 128 -> 256
        #     )
        # else:
        #     raise ValueError(f"Unsupported patch size: {self.patch_size_spatial}")

        # # Final projection to bands
        # final_ch = max(256, decoder_input_channels // 8)  # 256
        
        # self.decoder_out = nn.Sequential(
        #     nn.Conv2d(final_ch, final_ch, kernel_size=1), # Mixing
        #     nn.GELU(),
        #     nn.Conv2d(final_ch, num_bands, kernel_size=1) # Final Projection
        # )
        
        

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        This function takes a hyperspectral image (pixel_values) and reconstructs it.
        It does this by:
        1. Encoding the image into a sequence of tokens.
        2. Decoding the tokens into a reconstructed hyperspectral image.
        
        Inputs:
            pixel_values: hyperspectral batch tensor of shape (batch_size, num_bands, image_h, image_w)
                          For EMIT chips of size 256x256: (B, 285, 256, 256)
        Outputs:
            recon_pixel_values: reconstructed hyperspectral image of shape (batch_size, num_bands, image_h, image_w)
        """
        
        batch_size = pixel_values.shape[0]
        
        # 1. Encode: (B, 1 + Num_Tokens, D)
        # 1024 comes from 256 spatial patches * 4 spectral summaries
        tokens = self.hsi_encoder(pixel_values)
        
        # 2. Classifier Head: (B, 1, D) -> (B, 1)
        # Token 0 is the classifier summary
        #cls_token = tokens[:, 0] 
        # Tokens 1..End are the spatial/spectral data for reconstruction
        visual_tokens = tokens[:, 1:]
        
        # Run Classifier (Output shape: B, 1)
        #plume_logits = self.plume_classifier(cls_token)
        
        # 2. Take the MAX activation across all spatial locations.
        # This asks: "What is the strongest feature present ANYWHERE in this image?"
        # If a plume exists in patch #42, its features will survive the max pool.
        max_pooled_features, _ = torch.max(visual_tokens, dim=1) # Shape: (B, D)
        
        # 3. Classify based on that strongest feature
        plume_logits = self.plume_classifier(max_pooled_features)
        
        # 2. Reshape for Decoding
        # We separate spatial (256) from spectral (4)
        # Encoder returns tokens of shape: (B, num_patches * num_queries, D)
        # We want: (B, num_patches, num_queries, D)
        visual_tokens = visual_tokens.view(batch_size, -1, self.num_spectral_queries, self.hidden_size)

        # Flatten the spectral dimension into the feature dimension
        # We want one vector per spatial location that contains all spectral info
        # (B, 256, 4*D)
        visual_tokens = visual_tokens.flatten(2)
        
        # 3. Project to Decoder Channels
        # (B, 256, 4*D) -> (B, 256, 256_channels)
        dec_feats = self.decoder_projector(visual_tokens)
        
        # 4. Reshape to Grid for PixelShuffle
        patch_grid_h = self.hsi_encoder.grid_size
        patch_grid_w = self.hsi_encoder.grid_size
        dec_feats = dec_feats.transpose(1, 2).reshape(batch_size, -1, patch_grid_h, patch_grid_w)
        
        # 5. Spatial Upsampling
        # (B, 256, 16, 16) -> (B, 32, 256, 256)
        up_feats = self.decoder_ps(dec_feats)
        
        # 6. Final Spectral Projection
        reconstructed_hypercube = self.decoder_out(up_feats)
        
        return reconstructed_hypercube, plume_logits


## Dataloader definition
class EmitRadDataset(Dataset):
    def __init__(self, paths, band_mean, band_std, fill=-9999.0):
        self.paths = paths
        self.band_mean = torch.tensor(band_mean).view(-1, 1, 1)  # (C,1,1)
        self.band_std  = torch.tensor(band_std ).view(-1, 1, 1)  # (C,1,1)
        self.fill = fill

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        path = self.paths[i]
        x = np.load(path, mmap_mode="r")
        
        if x.dtype != np.float32:
            x = x.astype(np.float32, copy=False)
        x = torch.from_numpy(x).permute(2, 0, 1)  # (B,H,W)
        
        valid = (x != self.fill) & torch.isfinite(x) #Make a mask of valid pixels using EMIT's -9999 fill value  # (B,H,W)
        x = torch.where(valid, (x - self.band_mean) / self.band_std, torch.zeros_like(x))  #Do normalization
        
        # Check path string for keywords
        if "PlumeComplex" in path:
            label = 1.0
        elif "negative_chip" in path:
            label = 0.0
        else:
            # Fallback (safe default to 0.0 if naming convention fails)
            label = 0.0
        
        # Create float tensor for BCEWithLogitsLoss (shape: [1])
        label_t = torch.tensor([label], dtype=torch.float32)
        
        return x, valid, label_t

    # def __getitem__(self, i):
    #     x = np.load(self.paths[i], mmap_mode="r").astype(np.float32)  # (H,W,B)
    #     x = torch.from_numpy(x).permute(2, 0, 1)                     # (B,H,W)
    #     valid = (x != self.fill) & torch.isfinite(x) #Make a mask of valid pixels using EMIT's -9999 fill value  # (B,H,W)

    #     # normalize only valid pixels; put invalids at 0 after normalization
    #     x = torch.where(valid, (x - self.band_mean) / self.band_std, torch.zeros_like(x))
        
    #     return x, valid

### Dataset Normalization Classes:

def estimate_band_stats(paths, max_files=200):
    paths = paths[:max_files]
    sum_ = np.zeros(BANDS, dtype=np.float64)
    sum2 = np.zeros(BANDS, dtype=np.float64)
    cnt = np.zeros(BANDS, dtype=np.int64)

    for i, p in enumerate(paths):
        if i % 5 == 0:
            print(f"processing {i}/{len(paths)}")
        x = np.load(p, mmap_mode="r")                 # (H,W,B) float32
        #print(x.shape)
        assert x.shape[-1] == BANDS
        valid = (x != FILL) & np.isfinite(x)
        xv = np.where(valid, x, 0.0).astype(np.float64)
        sum_ += xv.sum(axis=(0,1))
        sum2 += (xv * xv).sum(axis=(0,1))
        cnt += valid.sum(axis=(0,1))

    mean = sum_ / np.maximum(cnt, 1)
    var = sum2 / np.maximum(cnt, 1) - mean**2
    std = np.sqrt(np.maximum(var, 1e-12))
    return mean.astype(np.float32), std.astype(np.float32), cnt

def _band_stats_worker(args):
    paths, fill, bands = args
    sum_ = np.zeros(bands, dtype=np.float64)
    sum2 = np.zeros(bands, dtype=np.float64)
    cnt = np.zeros(bands, dtype=np.int64)

    for p in paths:
        x = np.load(p, mmap_mode="r")  # (H,W,B)
        # safety: allow either (H,W,B) or (B,H,W) if you ever change formats
        if x.shape[-1] != bands:
            raise ValueError(f"{p}: expected last dim == {bands}, got {x.shape}")

        valid = (x != fill) & np.isfinite(x)
        xv = np.where(valid, x, 0.0).astype(np.float64)

        sum_ += xv.sum(axis=(0, 1))
        sum2 += (xv * xv).sum(axis=(0, 1))
        cnt += valid.sum(axis=(0, 1))

    return sum_, sum2, cnt

def estimate_band_stats_parallel(paths, max_files=None, num_workers=None, chunks_per_worker=4, fill=FILL, bands=BANDS):
    """
    Parallel band stats over many .npy cubes.

    Returns:
      band_mean: (bands,) float32
      band_std:  (bands,) float32
      band_cnt:  (bands,) int64
    """
    if max_files is not None:
        paths = paths[:max_files]
    if num_workers is None:
        num_workers = max(1, (os.cpu_count() or 1) - 1)

    # Split into moderately sized chunks to balance overhead vs load balancing
    n_chunks = max(num_workers * chunks_per_worker, 1)
    chunks = np.array_split(np.array(paths, dtype=object), n_chunks)
    worker_args = [(chunk.tolist(), fill, bands) for chunk in chunks if len(chunk) > 0]

    sum_total = np.zeros(bands, dtype=np.float64)
    sum2_total = np.zeros(bands, dtype=np.float64)
    cnt_total = np.zeros(bands, dtype=np.int64)

    with mp.get_context("fork").Pool(processes=num_workers) as pool:
        for sum_, sum2, cnt in pool.imap_unordered(_band_stats_worker, worker_args):
            sum_total += sum_
            sum2_total += sum2
            cnt_total += cnt

    mean = sum_total / np.maximum(cnt_total, 1)
    var = sum2_total / np.maximum(cnt_total, 1) - mean**2
    std = np.sqrt(np.maximum(var, 1e-12))
    return mean.astype(np.float32), std.astype(np.float32), cnt_total

def load_band_stats(path):
    d = np.load(path)
    band_mean = d["band_mean"].astype(np.float32)
    band_std  = d["band_std"].astype(np.float32)
    band_cnt  = d["band_cnt"].astype(np.int64) if "band_cnt" in d.files else None
    wavelengths_nm = d["wavelengths_nm"].astype(np.float32) if "wavelengths_nm" in d.files else None
    return band_mean, band_std, band_cnt, wavelengths_nm

def load_emit_wavelengths_nm(instrument_path=INSTRUMENT_PATH, expected_bands=BANDS):
    with open(instrument_path, "r", encoding="utf-8") as file_handle:
        instrument_metadata = json.load(file_handle)
    wavelengths_nm = np.asarray(instrument_metadata["wavelengths"], dtype=np.float32)
    if wavelengths_nm.ndim != 1 or wavelengths_nm.shape[0] != expected_bands:
        raise ValueError(f"Expected {expected_bands} wavelengths, got {wavelengths_nm.shape}")
    if not np.all(np.isfinite(wavelengths_nm)):
        raise ValueError("Non-finite wavelength values found in instrument metadata.")
    return wavelengths_nm

def save_band_stats(path, band_mean, band_std, band_cnt=None, wavelengths_nm=None, **meta):
    payload = {
        "band_mean": band_mean.astype(np.float32),
        "band_std":  band_std.astype(np.float32),
    }
    if band_cnt is not None:
        payload["band_cnt"] = band_cnt.astype(np.int64)
    if wavelengths_nm is not None:
        payload["wavelengths_nm"] = np.asarray(wavelengths_nm, dtype=np.float32)
    for k, v in meta.items():
        payload[k] = np.array(v)
    np.savez(path, **payload)



### Loss functions:

def masked_mse(x_hat, x, valid, eps=1e-6):
    # all tensors: (B, C, H, W)
    diff2 = (x_hat - x) ** 2
    diff2 = diff2 * valid
    denom = valid.sum().clamp_min(eps)
    return diff2.sum() / denom

def masked_band_mse(x_hat, x, valid, eps=1e-6):
    """
    Masked, *band-balanced* mean squared error for hyperspectral reconstruction.

    Purpose:
      Compute reconstruction error while ignoring invalid pixels (e.g., EMIT fill values)
      and ensuring that *each spectral band contributes equally* to the final loss.

    Rationale:
      A naive masked MSE averages over all valid pixels across all bands. That can let bands
      with more variance / larger dynamic range dominate the loss, causing the model to
      underfit "quieter" wavelengths. This loss computes an MSE *per band* (normalized by the
      number of valid pixels in that band) and then averages across bands, encouraging the
      decoder to reconstruct all wavelengths well.

    Args:
      x_hat:  reconstructed cube, shape (B, C, H, W)
      x:      target cube,        shape (B, C, H, W)
      valid:  mask in {0,1},      shape (B, C, H, W) where 1 indicates valid pixels
      eps:    small constant to avoid divide-by-zero when a band has no valid pixels

    Returns:
      Scalar loss (float tensor).
    """
    # x_hat, x, valid: (B, C, H, W)
    diff2 = (x_hat - x) ** 2
    diff2 = diff2 * valid
    denom_per_band = valid.sum(dim=(0, 2, 3)).clamp_min(eps)          # (C,)
    mse_per_band = diff2.sum(dim=(0, 2, 3)) / denom_per_band          # (C,)
    return mse_per_band.mean()

def masked_spectral_deriv_loss(x_hat, x, valid, eps=1e-6):
    """
    Masked spectral-derivative loss (adjacent-band slope matching).

    Purpose:
      Encourage the reconstruction to preserve *spectral shape* by matching the
      differences between adjacent wavelength bands, not just absolute band values.

    Rationale:
      In hyperspectral imagery, many downstream cues live in the *relative* spectrum
      (e.g., absorption features and smooth/structured changes across wavelength).
      A model can sometimes achieve good pixelwise MSE while still producing spectra that
      are "wiggly" or have distorted band-to-band transitions. By penalizing errors in
      adjacent-band differences (x[:, b+1] - x[:, b]), this loss stabilizes spectral shape
      and helps preserve physically meaningful signatures. The validity mask is applied
      to band pairs so we only score derivatives where *both* bands are valid.

    Args:
      x_hat: reconstructed cube, shape (B, C, H, W)
      x:     target cube,        shape (B, C, H, W)
      valid: mask in {0,1},      shape (B, C, H, W)
      eps:   small constant to avoid divide-by-zero when no valid band-pairs exist

    Returns:
      Scalar loss (float tensor).
    """
    # match adjacent-band differences; valid for both bands in the pair
    dx_hat = x_hat[:, 1:] - x_hat[:, :-1]                             # (B, C-1, H, W)
    dx = x[:, 1:] - x[:, :-1]
    valid_pair = valid[:, 1:] * valid[:, :-1]
    diff2 = (dx_hat - dx) ** 2
    diff2 = diff2 * valid_pair
    denom = valid_pair.sum().clamp_min(eps)
    return diff2.sum() / denom
  
#Discourage reconstructions that are all one intensity
def spatial_variance_loss(x_hat, x, valid, eps=1e-6):
    def spatial_grad(x):
        # x: (B,C,H,W)
        dx = x[..., 1:, :] - x[..., :-1, :]
        dy = x[..., :, 1:] - x[..., :, :-1]
        return dx, dy
    
    # spatial structure: match spatial gradients on valid pixels (use valid mask trimmed to match shapes)
    v = valid.float()
    dxh, dyh = spatial_grad(x_hat)
    dxt, dyt = spatial_grad(x)
    vx = v[..., 1:, :] * v[..., :-1, :]
    vy = v[..., :, 1:] * v[..., :, :-1]

    spatial = ((dxh - dxt).pow(2) * vx).sum() / vx.sum().clamp_min(eps) \
            + ((dyh - dyt).pow(2) * vy).sum() / vy.sum().clamp_min(eps)
            
    return spatial

def masked_sam_loss(x_hat, x, valid, eps=1e-8):
    """
    Masked Spectral Angle Mapper (SAM) Loss.
    
    Measures the angular difference between the reconstructed and target spectra.
    Crucial for identifying materials (and gases) independent of illumination/brightness.
    
    Formula: arccos( (x . x_hat) / (|x| * |x_hat|) )
    
    In hyperspectral remote sensing, the gold standard for comparing two spectra is SAM. It measures the angle between the reconstructed vector and the true vector, ignoring the magnitude (brightness).

    Scenario: A cloud shadow passes over the ground.

        MSE Loss: Huge error (pixel values changed from 100 to 50).

        SAM Loss: Zero error (the shape of the spectrum—sand is still sand—is identical).

    Scenario: A methane plume appears.

        MSE Loss: Tiny error (values changed from 50 to 48).

        SAM Loss: Large error (The absorption dip changes the vector direction).
    """
    # x_hat, x: (B, C, H, W) -> Permute to (B, H, W, C) for easier dot products
    # We want to treat the "C" dimension as the vector.
    
    # 1. Compute Dot Product along spectral dim
    dot_prod = (x_hat * x).sum(dim=1)  # (B, H, W)
    
    # 2. Compute Magnitudes (L2 Norm) along spectral dim
    norm_x = torch.norm(x, dim=1)      # (B, H, W)
    norm_hat = torch.norm(x_hat, dim=1) # (B, H, W)
    
    # 3. Compute Cosine Similarity
    # Clamp to [-1, 1] to avoid NaNs in acos due to float precision
    cos_sim = dot_prod / (norm_x * norm_hat).clamp_min(eps)
    cos_sim = cos_sim.clamp(-1.0 + eps, 1.0 - eps)
    
    # 4. Compute Spectral Angle (SAM)
    sam_angle = torch.acos(cos_sim) # Range [0, pi]
    
    # 5. Mask and Average
    # Valid mask is (B, C, H, W). We need a spatial mask (B, H, W).
    # A pixel is valid if ALL its bands are valid (or at least one? usually all).
    # Here we assume if the mask is 0, it's fill value.
    # We take valid[:, 0, :, :] assuming mask is same across bands for a pixel.
    spatial_mask = valid[:, 0, :, :]
    
    sam_angle = sam_angle * spatial_mask
    
    denom = spatial_mask.sum().clamp_min(eps)
    return sam_angle.sum() / denom

def compute_orthogonality_loss(query_tensor, device=None, epsilon=1e-8):
    """
    Computes an orthogonality penalty to force query vectors to be distinct.
    
    This prevents "Mode Collapse" where multiple attention heads or queries 
    converge to the same feature, effectively reducing the model's capacity.
    
    Args:
        query_tensor (torch.Tensor): The learnable query parameters. 
                                     Expected shapes: (N, Dim) or (1, 1, N, Dim).
        device (torch.device, optional): Device to create the identity matrix on. 
                                         If None, uses query_tensor.device.
        epsilon (float): Small constant for numerical stability during normalization.
        
    Returns:
        torch.Tensor: Scalar loss value.
    """
    # 1. Standardize Shape: Ensure we have (Num_Queries, Embed_Dim)
    # The model defines queries as (1, 1, num_queries, embed_dim)
    if query_tensor.dim() > 2:
        # Flatten all dimensions except the last one (Embed_Dim) 
        # to treat them as a bag of vectors
        queries = query_tensor.view(-1, query_tensor.size(-1)) 
    else:
        queries = query_tensor

    # 2. Normalize Vectors
    # We only care about the direction (angle), not magnitude.
    # L2 Normalization ensures dot product = cosine similarity.
    queries_norm = F.normalize(queries, p=2, dim=1, eps=epsilon)

    # 3. Compute Gram Matrix (Cosine Similarity Matrix)
    # Result shape: (Num_Queries, Num_Queries)
    # Entry [i, j] is the cosine similarity between query i and query j.
    gram_matrix = torch.mm(queries_norm, queries_norm.t())

    # 4. Create Target Identity Matrix
    # We want the diagonal to be 1 (self-similarity) and off-diagonal to be 0 (orthogonal).
    if device is None:
        device = query_tensor.device
    num_queries = queries.size(0)
    identity = torch.eye(num_queries, device=device)

    # 5. Compute MSE between Gram Matrix and Identity
    # This penalizes any non-zero value in the off-diagonal elements.
    ortho_loss = ((gram_matrix - identity) ** 2).sum() / (num_queries * num_queries)
    
    return ortho_loss

def compute_fft_loss(x_hat, x, valid, wn_start=5, wn_end=40, eps=1e-8):
    """
    Computes MSE between the Zonal Power Spectra of x and x_hat.
    Forces the model to match the energy distribution across spatial frequencies.
    
    Args:
        x_hat, x: (B, C, H, W)
        valid:    (B, C, H, W)
    """
    # 1. Mask the inputs to avoid edge artifacts from fill values
    x_masked = x * valid
    x_hat_masked = x_hat * valid

    # 2. Compute FFT along Longitude (Width, dim=-1)
    #    Use norm='ortho' to keep magnitudes consistent
    fft_x = torch.fft.rfft(x_masked, dim=-1, norm="ortho")
    fft_hat = torch.fft.rfft(x_hat_masked, dim=-1, norm="ortho")

    # 3. Compute Power Spectrum (Magnitude Squared)
    #    (B, C, H, W_freq)
    pow_x = fft_x.abs() ** 2
    pow_hat = fft_hat.abs() ** 2

    # 4. Compute Zonal Mean (Average along Latitude/Height, dim=-2)
    #    This creates the "Zonal Power Spectrum" used in the viz.
    #    Shape becomes: (B, C, W_freq)
    zonal_x = pow_x.mean(dim=-2)
    zonal_hat = pow_hat.mean(dim=-2)

    # 5. --- APPLY CUTOFFS ---
    # Slice the last dimension (frequency dimension)
    # If wn_end is None, it slices to the end
    target_x = zonal_x[..., wn_start:wn_end]
    target_hat = zonal_hat[..., wn_start:wn_end]

    # 6. Compute MSE on the sliced spectra
    loss = (target_hat - target_x).pow(2).mean()

    return loss
    

## Actual loss function used in model:
def recon_loss(x_hat, x, valid, alpha=0.1, beta=0.1, gamma=0.5, delta=0.1, theta=0.1, model=None, verbose=False, return_components=False, wn_start=5, wn_end=40):
    """
    Composite Loss Function for Hyperspectral Plume Reconstruction.
    
    Weights:
        Base (MSE):  1.0  - General fidelity.
        Alpha (Deriv): 0.1  - Preserves local shape (slopes).
        Beta (Spatial): 0.01 - Reduced! Only prevents extreme pixel noise.
        Gamma (SAM):   0.1  - Preserves chemical identity (vector direction).
    """
    # 1. Band-Balanced MSE (The Foundation)
    l_mse = masked_band_mse(x_hat, x, valid)
    
    # 2. Spectral Derivative (The Shape)
    l_deriv = masked_spectral_deriv_loss(x_hat, x, valid)
    
    # 3. Spectral Angle (The Chemistry) [NEW]
    l_sam = masked_sam_loss(x_hat, x, valid)
    
    # 4. Spatial Variance (The Texture)
    # KEPT LOW to avoid blurring out plumes.
    l_spatial = spatial_variance_loss(x_hat, x, valid)
    
    # 5. Orthogonality Loss (ensure spectral queries are distinct)
    l_ortho = 0.0
    if model is not None:
        # --- FIX 1: Handle DataParallel wrapping ---
        if hasattr(model, "module"):
            encoder = model.module.hsi_encoder
        else:
            encoder = model.hsi_encoder
            
        # --- FIX 2: Correct attribute name is 'hsi_encoder', not 'hyperspectral_encoder' ---
        if getattr(encoder, "spectral_queries", None) is not None:
            l_ortho = compute_orthogonality_loss(encoder.spectral_queries)
        else:
            # Fallback if variable names change in future
            print("Warning: Could not find 'spectral_queries' in model for orthogonality loss.")
            
    elif delta > 0 and verbose:
        # Only warn if we actually expected to calculate this loss
        print("Warning: Orthogonality loss weight > 0 but no model passed to loss function.")
        
    # 6. FFT Loss:
    l_fft = compute_fft_loss(x_hat, x, valid, wn_start=wn_start, wn_end=wn_end)
        
    if verbose: print(f"l_mse: {l_mse}, l_deriv: {l_deriv}, l_sam: {l_sam}, l_spatial: {l_spatial}, l_ortho: {l_ortho}, l_fft: {l_fft}")
    
    total_loss = l_mse + (alpha * l_deriv) + (beta * l_spatial) + (gamma * l_sam) + (delta * l_ortho) + (theta * l_fft)
    if return_components:
        return total_loss, {
            "l_mse": l_mse,
            "l_deriv": l_deriv,
            "l_sam": l_sam,
            "l_spatial": l_spatial,
            "l_ortho": l_ortho,
            "l_fft": l_fft}
    else:
        return total_loss
    
def do_classification(logits, y, threshold=0.5):
    num_tp = 0
    num_fp = 0
    num_tn = 0
    num_fn = 0
    accuracy = 0.0
    
    probs = torch.sigmoid(logits)
    y_hat = (probs > threshold).float()
    
    num_tp += ((y_hat == 1) & (y == 1)).sum().item()
    num_tn += ((y_hat == 0) & (y == 0)).sum().item()
    num_fp += ((y_hat == 1) & (y == 0)).sum().item()
    num_fn += ((y_hat == 0) & (y == 1)).sum().item()
    
    total = (num_tp + num_tn + num_fp + num_fn)
    if total > 0:
        accuracy = (num_tp + num_tn) / total
        
    return accuracy, num_tp, num_fp, num_tn, num_fn

def class_loss(plume_logits, label):
    return F.binary_cross_entropy_with_logits(plume_logits, label)

