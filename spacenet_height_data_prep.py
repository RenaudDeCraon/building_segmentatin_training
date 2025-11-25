#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SpaceNet Data Preparation for Building Height Estimation
Memory-efficient version - processes statistics incrementally
"""
import rasterio
from rasterio.plot import reshape_as_image
import numpy as np
from pathlib import Path
import cv2
from sklearn.model_selection import train_test_split
import json
from tqdm import tqdm
import gc

class SpaceNetHeightPreparation:
    def __init__(self, rgb_dir, truth_dir, output_dir, img_size=(256, 256)):
        self.rgb_dir = Path(rgb_dir)
        self.truth_dir = Path(truth_dir)
        self.output_dir = Path(output_dir)
        self.img_size = img_size
        
        # Create output directories
        self.images_dir = self.output_dir / 'images'
        self.masks_dir = self.output_dir / 'masks'
        self.heights_dir = self.output_dir / 'heights'
        
        for dir_path in [self.images_dir, self.masks_dir, self.heights_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Statistics for normalization (using Welford's online algorithm)
        self.height_stats = {
            'min': float('inf'),
            'max': float('-inf'),
            'mean': 0.0,
            'M2': 0.0,  # For variance calculation
            'count': 0
        }
    
    def find_matching_files(self, rgb_file):
        """Find CLS and AGL files matching an RGB file"""
        rgb_base = rgb_file.stem.replace('_RGB', '')
        
        cls_files = list(self.truth_dir.glob(f"*{rgb_base}*CLS.tif"))
        agl_files = list(self.truth_dir.glob(f"*{rgb_base}*AGL.tif"))
        
        cls_file = cls_files[0] if cls_files else None
        agl_file = agl_files[0] if agl_files else None
        
        return cls_file, agl_file
    
    def process_building_mask(self, cls_data):
        """Process CLS data to create binary building mask"""
        building_mask = (cls_data >= 5).astype(np.float32)
        return building_mask
    
    def update_statistics(self, building_heights):
        """
        Update statistics using Welford's online algorithm
        Memory efficient - doesn't store all values
        """
        for height in building_heights.flatten():
            self.height_stats['count'] += 1
            self.height_stats['min'] = min(self.height_stats['min'], float(height))
            self.height_stats['max'] = max(self.height_stats['max'], float(height))
            
            # Welford's algorithm for running mean and variance
            delta = height - self.height_stats['mean']
            self.height_stats['mean'] += delta / self.height_stats['count']
            delta2 = height - self.height_stats['mean']
            self.height_stats['M2'] += delta * delta2
    
    def process_height_data(self, agl_data, building_mask):
        """Process AGL data and update statistics"""
        height_map = agl_data.astype(np.float32)
        height_map = height_map * building_mask
        
        # Update statistics (only for building pixels)
        building_heights = height_map[building_mask > 0]
        if len(building_heights) > 0:
            self.update_statistics(building_heights)
        
        return height_map
    
    def resize_data(self, image, mask, height_map):
        """Resize all data to target size"""
        image_resized = cv2.resize(image, self.img_size, interpolation=cv2.INTER_LINEAR)
        mask_resized = cv2.resize(mask, self.img_size, interpolation=cv2.INTER_NEAREST)
        height_resized = cv2.resize(height_map, self.img_size, interpolation=cv2.INTER_LINEAR)
        
        return image_resized, mask_resized, height_resized
    
    def process_sample(self, rgb_file, output_id):
        """Process a single RGB-CLS-AGL triplet"""
        try:
            cls_file, agl_file = self.find_matching_files(rgb_file)
            
            if cls_file is None or agl_file is None:
                return None
            
            # Load RGB image
            with rasterio.open(rgb_file) as src:
                rgb_data = src.read()
                rgb_img = reshape_as_image(rgb_data)
                
                if rgb_img.dtype != np.uint8:
                    rgb_img = ((rgb_img - rgb_img.min()) / 
                              (rgb_img.max() - rgb_img.min()) * 255).astype(np.uint8)
            
            # Load CLS (building mask)
            with rasterio.open(cls_file) as src:
                cls_data = src.read(1)
                building_mask = self.process_building_mask(cls_data)
            
            # Load AGL (height data)
            with rasterio.open(agl_file) as src:
                agl_data = src.read(1)
                height_map = self.process_height_data(agl_data, building_mask)
            
            # Check if there are any buildings
            if building_mask.sum() == 0:
                return None
            
            # Resize all data
            rgb_resized, mask_resized, height_resized = self.resize_data(
                rgb_img, building_mask, height_map
            )
            
            # Save files
            image_path = self.images_dir / f'{output_id:06d}.png'
            mask_path = self.masks_dir / f'{output_id:06d}.png'
            height_path = self.heights_dir / f'{output_id:06d}.npy'
            
            cv2.imwrite(str(image_path), cv2.cvtColor(rgb_resized, cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(mask_path), (mask_resized * 255).astype(np.uint8))
            np.save(str(height_path), height_resized)
            
            # Clean up large arrays
            del rgb_data, rgb_img, cls_data, building_mask, agl_data, height_map
            del rgb_resized, mask_resized, height_resized
            
            return {
                'image': str(image_path),
                'mask': str(mask_path),
                'height': str(height_path),
            }
            
        except Exception as e:
            print(f"Error processing {rgb_file.name}: {e}")
            return None
    
    def prepare_dataset(self, val_split=0.15, test_split=0.1):
        """Process all RGB files and create train/val/test splits"""
        print("🚀 Starting SpaceNet Height Estimation Data Preparation")
        print("=" * 60)
        
        rgb_files = list(self.rgb_dir.glob("*.tif"))
        print(f"Found {len(rgb_files)} RGB files")
        
        processed_samples = []
        output_id = 0
        
        for idx, rgb_file in enumerate(tqdm(rgb_files, desc="Processing samples")):
            result = self.process_sample(rgb_file, output_id)
            if result is not None:
                processed_samples.append(result)
                output_id += 1
            
            # Garbage collection every 100 files
            if idx % 100 == 0:
                gc.collect()
        
        print(f"\n✅ Successfully processed {len(processed_samples)} samples")
        
        # Calculate final statistics
        if self.height_stats['count'] > 1:
            variance = self.height_stats['M2'] / (self.height_stats['count'] - 1)
            std = np.sqrt(variance)
        else:
            std = 0.0
        
        print("\n📊 Height Statistics:")
        print(f"  Min height: {self.height_stats['min']:.2f}m")
        print(f"  Max height: {self.height_stats['max']:.2f}m")
        print(f"  Mean height: {self.height_stats['mean']:.2f}m")
        print(f"  Std height: {std:.2f}m")
        print(f"  Total building pixels: {self.height_stats['count']}")
        
        # Save statistics
        stats_to_save = {
            'min': float(self.height_stats['min']) if self.height_stats['min'] != float('inf') else 0.0,
            'max': float(self.height_stats['max']) if self.height_stats['max'] != float('-inf') else 0.0,
            'mean': float(self.height_stats['mean']),
            'std': float(std),
            'samples': int(self.height_stats['count'])
        }
        
        stats_path = self.output_dir / 'height_statistics.json'
        with open(stats_path, 'w') as f:
            json.dump(stats_to_save, f, indent=2)
        print(f"\n💾 Statistics saved to {stats_path}")
        
        # Split dataset
        train_val, test = train_test_split(
            processed_samples, 
            test_size=test_split, 
            random_state=42
        )
        
        train, val = train_test_split(
            train_val, 
            test_size=val_split/(1-test_split), 
            random_state=42
        )
        
        print(f"\n📂 Dataset splits:")
        print(f"  Training: {len(train)} samples")
        print(f"  Validation: {len(val)} samples")
        print(f"  Test: {len(test)} samples")
        
        # Save split information
        splits = {
            'train': train,
            'val': val,
            'test': test
        }
        
        for split_name, split_data in splits.items():
            split_file = self.output_dir / f'{split_name}_data.json'
            with open(split_file, 'w') as f:
                json.dump(split_data, f, indent=2)
            
            images_file = self.output_dir / f'{split_name}_images.txt'
            masks_file = self.output_dir / f'{split_name}_masks.txt'
            heights_file = self.output_dir / f'{split_name}_heights.txt'
            
            with open(images_file, 'w') as f:
                f.write('\n'.join([s['image'] for s in split_data]))
            
            with open(masks_file, 'w') as f:
                f.write('\n'.join([s['mask'] for s in split_data]))
            
            with open(heights_file, 'w') as f:
                f.write('\n'.join([s['height'] for s in split_data]))
        
        print(f"\n✅ Data preparation complete!")
        print(f"📁 Output directory: {self.output_dir}")
        
        return processed_samples

def main():
    RGB_DIR = "/Users/umutmurat/Documents/Data/Track1-RGB"
    TRUTH_DIR = "/Users/umutmurat/Documents/Data/Track1-Truth 2"
    OUTPUT_DIR = "spacenet_height_dataset"
    IMG_SIZE = (256, 256)
    
    preparator = SpaceNetHeightPreparation(
        rgb_dir=RGB_DIR,
        truth_dir=TRUTH_DIR,
        output_dir=OUTPUT_DIR,
        img_size=IMG_SIZE
    )
    
    samples = preparator.prepare_dataset(val_split=0.15, test_split=0.1)
    
    print("\n🎉 Dataset ready for height estimation training!")


if __name__ == "__main__":
    main()
