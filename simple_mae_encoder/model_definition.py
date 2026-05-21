### Purpose: Define a simple ViT encoder for hyperspectral images, pretrained using MAE masking

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import h5py


class SimpleMAEEncoder(nn.Module):
    def __init__(self, in_channels, out_channels, num_heads, num_layers):
        super(SimpleMAEEncoder, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_heads = num_heads
        self.num_layers = num_layers
        
    def forward(self, x):
        return x