#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 24 01:12:42 2025

@author: umutmurat
"""

"""
Visualize sample image-mask pairs to understand the data
"""
import rasterio
from rasterio.plot import reshape_as_image
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def visualize_samples(rgb_dir, truth_dir, num_samples=3):
    """
    Visualize sample RGB images with their corresponding masks
    """
    rgb_path = Path(rgb_dir)
    truth_path = Path(truth_dir)
    
    # Get sample RGB files
    rgb_files = list(rgb_path.glob("*.tif"))[:num_samples]
    
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5*num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i, rgb_file in enumerate(rgb_files):
        print(f"Processing {rgb_file.name}...")
        
        # Load RGB image
        with rasterio.open(rgb_file) as src:
            rgb_data = src.read()
            rgb_img = reshape_as_image(rgb_data)
            
            # Normalize if needed
            if rgb_img.dtype != np.uint8:
                rgb_img = ((rgb_img - rgb_img.min()) / (rgb_img.max() - rgb_img.min()) * 255).astype(np.uint8)
        
        # Find corresponding truth files
        rgb_base = rgb_file.stem.replace('_RGB', '')
        
        # Look for CLS file (building classification)
        cls_files = list(truth_path.glob(f"*{rgb_base}*CLS.tif"))
        agl_files = list(truth_path.glob(f"*{rgb_base}*AGL.tif"))
        
        # Display RGB image
        axes[i, 0].imshow(rgb_img)
        axes[i, 0].set_title(f'RGB Image\n{rgb_file.name}')
        axes[i, 0].axis('off')
        
        # Display building mask (CLS)
        if cls_files:
            with rasterio.open(cls_files[0]) as src:
                mask_data = src.read(1)
                
                # Convert to binary if needed
                if mask_data.max() > 1:
                    mask_binary = (mask_data > 0).astype(np.uint8)
                else:
                    mask_binary = mask_data.astype(np.uint8)
                
                axes[i, 1].imshow(mask_binary, cmap='gray')
                axes[i, 1].set_title(f'Building Mask (CLS)\nBuildings: {np.sum(mask_binary)} pixels')
                axes[i, 1].axis('off')
                
                print(f"  CLS mask shape: {mask_data.shape}")
                print(f"  CLS unique values: {np.unique(mask_data)}")
                print(f"  Building pixels: {np.sum(mask_binary)} / {mask_binary.size}")
        else:
            axes[i, 1].text(0.5, 0.5, 'No CLS file found', ha='center', va='center')
            axes[i, 1].set_title('Building Mask (CLS)\nNot Found')
            axes[i, 1].axis('off')
        
        # Display height data (AGL) if available
        if agl_files:
            with rasterio.open(agl_files[0]) as src:
                height_data = src.read(1)
                
                axes[i, 2].imshow(height_data, cmap='viridis')
                axes[i, 2].set_title(f'Height Data (AGL)\nMax: {height_data.max():.1f}')
                axes[i, 2].axis('off')
                
                print(f"  AGL height range: {height_data.min():.1f} - {height_data.max():.1f}")
        else:
            axes[i, 2].text(0.5, 0.5, 'No AGL file found', ha='center', va='center')
            axes[i, 2].set_title('Height Data (AGL)\nNot Found')
            axes[i, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig('sample_visualization.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"\n📊 Visualization saved as 'sample_visualization.png'")

if __name__ == "__main__":
    rgb_dir = "/Users/umutmurat/Documents/Data/Track1-RGB"
    truth_dir = "/Users/umutmurat/Documents/Data/Track1-Truth 2"
    
    visualize_samples(rgb_dir, truth_dir, num_samples=3)