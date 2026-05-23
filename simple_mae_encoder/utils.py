import numpy as np
import torch
import matplotlib.pyplot as plt


def visualize_hypercube(hypercube, bands=[36, 23, 9]):
    """
    Visualize a hypercube.  Assuming dims are (H, W, C) , (128x128x285)
    """
    #check if bands is an int:
    if isinstance(bands, int):
        bands = [bands, bands, bands]
        
    #slice hypercube using bands:
    hypercube = hypercube[:, :, bands]
    
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
    
    #visualize reconstruction and original side by side in one fig:
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(reconstruction)
    axes[0].set_title("Reconstruction")
    axes[1].imshow(original)
    axes[1].set_title("Original")
    plt.show()