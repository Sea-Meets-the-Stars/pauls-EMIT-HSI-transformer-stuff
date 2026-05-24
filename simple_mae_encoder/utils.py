import numpy as np
import torch
import matplotlib.pyplot as plt

import os
import glob
from torch.utils.data import Dataset, DataLoader
import h5py


def visualize_hypercube(hypercube, bands=[36, 23, 9]):
    """
    Visualize a hypercube.  Assuming dims are (H, W, C) , (128x128x285)
    """
    #check if bands is an int:
    if isinstance(bands, int):
        bands = [bands, bands, bands]
        
    #slice hypercube using bands:
    hypercube = hypercube[:, :, bands]
    
    #normalize hypercube to 0-1
    hypercube = (hypercube - np.min(hypercube)) / (np.max(hypercube) - np.min(hypercube))
    
    plt.imshow(hypercube)
    plt.show()
    

def visualize_reconstruction(reconstruction, original, bands=[36, 23, 9]):
    """
    Visualize a reconstruction.  Assuming dims are (H, W, C) , (128x128x285)
    """
    #check if bands is an int:
    if isinstance(bands, int):
        bands = [bands, bands, bands]
        
    #slice reconstruction and original using bands:
    reconstruction = reconstruction[:, :, bands]
    original = original[:, :, bands]
    
    #normalize reconstruction and original to 0-1
    original_min = np.min(original)
    original_max = np.max(original)
    original_range = original_max - original_min
    
    reconstruction_min = np.min(reconstruction)
    reconstruction_max = np.max(reconstruction)
    reconstruction_range = reconstruction_max - reconstruction_min
    
    original = (original - original_min) / original_range
    reconstruction_normed_to_original = (reconstruction - original_min) / original_range
    reconstruction = (reconstruction - reconstruction_min) / reconstruction_range
    
    #handle out of bounds pixel values:
    reconstruction_normed_to_original = np.clip(reconstruction_normed_to_original, 0, 1)
    
    #visualize reconstruction and original side by side in one fig:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(reconstruction)
    axes[0].set_title("Reconstruction")
    axes[1].imshow(original)
    axes[1].set_title("Original")
    axes[2].imshow(reconstruction_normed_to_original)
    axes[2].set_title("Reconstruction Normed to Original")
    
    plt.tight_layout()
    plt.show()

def normalize_chip(chip):
    chip_mean = chip.mean(axis=(0,1), keepdims=True)
    chip_std = chip.std(axis=(0,1), keepdims=True)
    return (chip - chip_mean) / (chip_std + 1e-6), chip_mean, chip_std

def unnormalize_chip(chip, chip_mean, chip_std):
    return chip * chip_std + chip_mean
    
class HyperspectralDataset(Dataset):
    def __init__(self, data_dir, max_files=None):
        """
        Args:
            data_dir (str): Path to the directory containing .h5 files.
            max_files (int): Limit the dataset size for overfitting tests.
        """
        # Find all .h5 files in the directory
        self.file_paths = glob.glob(os.path.join(data_dir, "*.h5"))
        
        # Limit to 100 files for your overfitting test
        if max_files is not None:
            self.file_paths = self.file_paths[:max_files]
        
        print(f"Dataset initialized with {len(self.file_paths)} files.")

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx, return_stats=False):
        # 1. Load the HDF5 file
        with h5py.File(self.file_paths[idx], 'r') as f:
            array = np.array(f['hypercube'], dtype=np.float32)
            
        # 2. Normalize the chip
        array, mean, std = normalize_chip(array)
        
        # 3. Return as PyTorch Tensor
        if return_stats:
            return torch.from_numpy(array), mean, std
        else:
            return torch.from_numpy(array)