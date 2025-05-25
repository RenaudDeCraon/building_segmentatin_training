#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 24 01:15:32 2025

@author: umutmurat
"""

"""
SpaceNet building dataset preparation for training
"""
import os
import numpy as np
import rasterio
from rasterio.plot import reshape_as_image
import cv2
from pathlib import Path
import shutil
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tqdm import tqdm
import albumentations as A

class SpaceNetDataPreparator:
    def __init__(self, rgb_dir, truth_dir, output_dir="spacenet_dataset", building_classes=None):
        self.rgb_dir = Path(rgb_dir)
        self.truth_dir = Path(truth_dir)
        self.output_dir = Path(output_dir)
        
        # Define which classes are buildings (adjust based on your CLS analysis)
        # For SpaceNet, typically classes >= 5 are buildings
        self.building_classes = building_classes or [5, 6, 9, 17, 65]
        
        # Create output directories
        self.dirs = {
            'train_images': self.output_dir / 'train' / 'images',
            'train_masks': self.output_dir / 'train' / 'masks',
            'val_images': self.output_dir / 'val' / 'images',
            'val_masks': self.output_dir / 'val' / 'masks',
            'test_images': self.output_dir / 'test' / 'images',
            'test_masks': self.output_dir / 'test' / 'masks'
        }
        
        for dir_path in self.dirs.values():
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def convert_cls_to_binary_mask(self, cls_array):
        """
        Convert CLS categorical mask to binary building mask
        """
        binary_mask = np.zeros_like(cls_array, dtype=np.uint8)
        
        # Set building pixels to 1
        for building_class in self.building_classes:
            binary_mask[cls_array == building_class] = 1
        
        return binary_mask
    
    def load_and_process_image_pair(self, rgb_file):
        """
        Load RGB image and corresponding CLS mask, process them
        """
        # Get base name for matching
        base_name = rgb_file.stem.replace('_RGB', '')
        
        # Find corresponding CLS file
        cls_file = self.truth_dir / f"{base_name}_CLS.tif"
        
        if not cls_file.exists():
            print(f"Warning: No CLS file found for {rgb_file.name}")
            return None, None
        
        try:
            # Load RGB image
            with rasterio.open(rgb_file) as src:
                rgb_data = src.read()
                rgb_img = reshape_as_image(rgb_data)
                
                # Handle different data types
                if rgb_img.dtype != np.uint8:
                    # Normalize to 0-255
                    rgb_img = ((rgb_img - rgb_img.min()) / (rgb_img.max() - rgb_img.min()) * 255).astype(np.uint8)
                
                # Ensure 3 channels
                if rgb_img.shape[2] > 3:
                    rgb_img = rgb_img[:, :, :3]
                elif rgb_img.shape[2] == 1:
                    rgb_img = np.repeat(rgb_img, 3, axis=2)
            
            # Load CLS mask
            with rasterio.open(cls_file) as src:
                cls_data = src.read(1)
                
                # Convert to binary mask
                binary_mask = self.convert_cls_to_binary_mask(cls_data)
            
            return rgb_img, binary_mask
            
        except Exception as e:
            print(f"Error processing {rgb_file.name}: {e}")
            return None, None
    
    def tile_image_pair(self, rgb_img, mask, tile_size=512, overlap=64, min_building_ratio=0.001):
        """
        Split large images into training tiles
        """
        tiles = []
        height, width = rgb_img.shape[:2]
        
        # Calculate tile positions
        y_positions = range(0, height - tile_size + 1, tile_size - overlap)
        x_positions = range(0, width - tile_size + 1, tile_size - overlap)
        
        # Add edge tiles
        if y_positions and y_positions[-1] + tile_size < height:
            y_positions = list(y_positions) + [height - tile_size]
        if x_positions and x_positions[-1] + tile_size < width:
            x_positions = list(x_positions) + [width - tile_size]
        
        for i, y in enumerate(y_positions):
            for j, x in enumerate(x_positions):
                # Extract tile
                rgb_tile = rgb_img[y:y+tile_size, x:x+tile_size]
                mask_tile = mask[y:y+tile_size, x:x+tile_size]
                
                # Pad if necessary
                if rgb_tile.shape[0] != tile_size or rgb_tile.shape[1] != tile_size:
                    rgb_tile = np.pad(rgb_tile, 
                                    ((0, tile_size - rgb_tile.shape[0]), 
                                     (0, tile_size - rgb_tile.shape[1]), 
                                     (0, 0)), mode='reflect')
                    mask_tile = np.pad(mask_tile,
                                     ((0, tile_size - mask_tile.shape[0]), 
                                      (0, tile_size - mask_tile.shape[1])), mode='reflect')
                
                # Calculate building ratio
                building_pixels = np.sum(mask_tile)
                total_pixels = tile_size * tile_size
                building_ratio = building_pixels / total_pixels
                
                # Keep tiles with minimum building content or some background tiles
                if building_ratio >= min_building_ratio or (building_ratio == 0 and np.random.random() < 0.1):
                    tiles.append({
                        'rgb': rgb_tile,
                        'mask': mask_tile,
                        'building_ratio': building_ratio,
                        'position': (x, y)
                    })
        
        return tiles
    
    def augment_tile(self, rgb_tile, mask_tile):
        """
        Apply data augmentation to a tile
        """
        transform = A.Compose([
            A.RandomRotate90(p=0.5),
            A.HorizontalFlip(p=0.5),
A.VerticalFlip(p=0.5),
            A.OneOf([
                A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2),
                A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8)),
            ], p=0.7),
            A.OneOf([
                A.Blur(blur_limit=3),
                A.GaussNoise(var_limit=25),
                A.RandomGamma(gamma_limit=(80, 120)),
            ], p=0.3),
        ])
        
        transformed = transform(image=rgb_tile, mask=mask_tile)
        return transformed['image'], transformed['mask']
    
    def process_all_images(self, tile_size=512, overlap=64, augment_factor=2, 
                          val_split=0.2, test_split=0.1, max_files=None):
        """
        Process all RGB images and create training dataset
        """
        rgb_files = list(self.rgb_dir.glob("*_RGB.tif"))
        
        if max_files:
            rgb_files = rgb_files[:max_files]
        
        print(f"Processing {len(rgb_files)} RGB images...")
        
        all_tiles = []
        processed_count = 0
        
        for rgb_file in tqdm(rgb_files, desc="Processing images"):
            rgb_img, mask = self.load_and_process_image_pair(rgb_file)
            
            if rgb_img is None or mask is None:
                continue
            
            # Create tiles
            tiles = self.tile_image_pair(rgb_img, mask, tile_size, overlap)
            
            # Add original tiles
            for tile_idx, tile in enumerate(tiles):
                tile_id = f"{rgb_file.stem}_{tile_idx}"
                all_tiles.append({
                    'rgb': tile['rgb'],
                    'mask': tile['mask'],
                    'id': tile_id,
                    'source_file': rgb_file.name,
                    'building_ratio': tile['building_ratio'],
                    'augmented': False
                })
            
            # Add augmented tiles
            for aug_idx in range(augment_factor):
                for tile_idx, tile in enumerate(tiles):
                    # Only augment tiles with buildings
                    if tile['building_ratio'] > 0:
                        aug_rgb, aug_mask = self.augment_tile(tile['rgb'], tile['mask'])
                        tile_id = f"{rgb_file.stem}_{tile_idx}_aug_{aug_idx}"
                        all_tiles.append({
                            'rgb': aug_rgb,
                            'mask': aug_mask,
                            'id': tile_id,
                            'source_file': rgb_file.name,
                            'building_ratio': tile['building_ratio'],
                            'augmented': True
                        })
            
            processed_count += 1
            
            # Progress update
            if processed_count % 100 == 0:
                print(f"Processed {processed_count}/{len(rgb_files)} images, "
                      f"Created {len(all_tiles)} tiles")
        
        print(f"\nTotal tiles created: {len(all_tiles)}")
        
        # Filter tiles by building content
        building_tiles = [t for t in all_tiles if t['building_ratio'] > 0]
        background_tiles = [t for t in all_tiles if t['building_ratio'] == 0]
        
        # Keep balanced ratio (e.g., 70% building tiles, 30% background)
        max_background = int(len(building_tiles) * 0.4)
        if len(background_tiles) > max_background:
            background_tiles = np.random.choice(background_tiles, max_background, replace=False).tolist()
        
        final_tiles = building_tiles + background_tiles
        np.random.shuffle(final_tiles)
        
        print(f"Final dataset: {len(final_tiles)} tiles")
        print(f"  Building tiles: {len(building_tiles)}")
        print(f"  Background tiles: {len(background_tiles)}")
        
        # Split dataset
        train_tiles, temp_tiles = train_test_split(final_tiles, 
                                                  test_size=(val_split + test_split), 
                                                  random_state=42)
        val_tiles, test_tiles = train_test_split(temp_tiles, 
                                                test_size=(test_split/(val_split + test_split)), 
                                                random_state=42)
        
        # Save tiles
        self.save_tiles(train_tiles, 'train')
        self.save_tiles(val_tiles, 'val')
        self.save_tiles(test_tiles, 'test')
        
        # Create file lists
        self.create_file_lists()
        
        # Print statistics
        self.print_dataset_stats(train_tiles, val_tiles, test_tiles)
        
        # Create sample visualizations
        self.create_sample_visualization(train_tiles[:9])
        
        return len(train_tiles), len(val_tiles), len(test_tiles)
    
    def save_tiles(self, tiles, split):
        """Save tiles to disk"""
        for tile in tqdm(tiles, desc=f"Saving {split} tiles"):
            # Save image
            img_path = self.dirs[f'{split}_images'] / f"{tile['id']}.png"
            cv2.imwrite(str(img_path), cv2.cvtColor(tile['rgb'], cv2.COLOR_RGB2BGR))
            
            # Save mask
            mask_path = self.dirs[f'{split}_masks'] / f"{tile['id']}.png"
            # Convert binary mask to 0-255 range
            mask_255 = (tile['mask'] * 255).astype(np.uint8)
            cv2.imwrite(str(mask_path), mask_255)
    
    def create_file_lists(self):
        """Create text files with paths for training script"""
        for split in ['train', 'val', 'test']:
            img_dir = self.dirs[f'{split}_images']
            mask_dir = self.dirs[f'{split}_masks']
            
            img_files = sorted(img_dir.glob("*.png"))
            mask_files = sorted(mask_dir.glob("*.png"))
            
            # Write image paths
            with open(self.output_dir / f'{split}_images.txt', 'w') as f:
                for img_file in img_files:
                    f.write(str(img_file) + '\n')
            
            # Write mask paths
            with open(self.output_dir / f'{split}_masks.txt', 'w') as f:
                for mask_file in mask_files:
                    f.write(str(mask_file) + '\n')
    
    def print_dataset_stats(self, train_tiles, val_tiles, test_tiles):
        """Print dataset statistics"""
        print(f"\n📊 DATASET STATISTICS")
        print("=" * 50)
        print(f"Training tiles: {len(train_tiles)}")
        print(f"Validation tiles: {len(val_tiles)}")
        print(f"Test tiles: {len(test_tiles)}")
        print(f"Total tiles: {len(train_tiles) + len(val_tiles) + len(test_tiles)}")
        
        # Building ratio statistics
        for split_name, tiles in [("Train", train_tiles), ("Val", val_tiles), ("Test", test_tiles)]:
            building_ratios = [t['building_ratio'] for t in tiles]
            tiles_with_buildings = sum(1 for r in building_ratios if r > 0)
            avg_building_ratio = np.mean([r for r in building_ratios if r > 0]) if tiles_with_buildings > 0 else 0
            
            print(f"\n{split_name} set:")
            print(f"  Tiles with buildings: {tiles_with_buildings}/{len(tiles)} ({100*tiles_with_buildings/len(tiles):.1f}%)")
            print(f"  Avg building ratio: {avg_building_ratio:.3f}")
    
    def create_sample_visualization(self, sample_tiles):
        """Create visualization of sample tiles"""
        fig, axes = plt.subplots(3, 6, figsize=(18, 9))
        
        for i, tile in enumerate(sample_tiles):
            row = i // 3
            col = (i % 3) * 2
            
            # Show image
            axes[row, col].imshow(tile['rgb'])
            axes[row, col].set_title(f'Image\nBuildings: {tile["building_ratio"]:.1%}')
            axes[row, col].axis('off')
            
            # Show mask
            axes[row, col + 1].imshow(tile['mask'], cmap='gray')
            axes[row, col + 1].set_title(f'Mask\n{tile["id"][:15]}...')
            axes[row, col + 1].axis('off')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'sample_tiles.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        print(f"Sample visualization saved to {self.output_dir / 'sample_tiles.png'}")

def main():
    # Configuration
    rgb_dir = "/Users/umutmurat/Documents/Data/Track1-RGB"
    truth_dir = "/Users/umutmurat/Documents/Data/Track1-Truth 2"
    output_dir = "spacenet_building_dataset"
    
    # Building classes (adjust based on your CLS analysis)
    # For SpaceNet, typically: background=2, buildings=5,6,9,17,65
    building_classes = [5, 6, 9, 17, 65]
    
    # Create preparator
    preparator = SpaceNetDataPreparator(
        rgb_dir=rgb_dir,
        truth_dir=truth_dir,
        output_dir=output_dir,
        building_classes=building_classes
    )
    
    print("🚀 Starting SpaceNet dataset preparation...")
    print(f"Building classes: {building_classes}")
    
    # Process dataset
    # Start with a smaller subset for testing
    train_count, val_count, test_count = preparator.process_all_images(
        tile_size=512,
        overlap=64,
        augment_factor=2,
        max_files=100  # Remove this to process all files
    )
    
    print(f"\n✅ Dataset preparation complete!")
    print(f"📁 Output directory: {output_dir}")
    print(f"📊 Ready for training with {train_count} training tiles")

if __name__ == "__main__":
    main()