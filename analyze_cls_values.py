#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 24 01:14:46 2025

@author: umutmurat
"""

"""
Analyze CLS values to understand which ones represent buildings
"""
import rasterio
import numpy as np
from pathlib import Path
from collections import Counter
import matplotlib.pyplot as plt

def analyze_cls_values(truth_dir, sample_size=20):
    """
    Analyze CLS file values to determine building class codes
    """
    truth_path = Path(truth_dir)
    cls_files = list(truth_path.glob("*CLS.tif"))
    
    print("🔍 ANALYZING CLS VALUES")
    print("=" * 50)
    
    all_values = []
    file_stats = []
    
    # Sample files to analyze
    sample_files = cls_files[:sample_size]
    
    for i, cls_file in enumerate(sample_files):
        try:
            with rasterio.open(cls_file) as src:
                data = src.read(1)
                unique_vals, counts = np.unique(data, return_counts=True)
                
                print(f"\nFile {i+1}: {cls_file.name}")
                print(f"  Shape: {data.shape}")
                print(f"  Classes found: {unique_vals}")
                
                # Calculate percentages
                total_pixels = data.size
                for val, count in zip(unique_vals, counts):
                    percentage = (count / total_pixels) * 100
                    print(f"    Class {val}: {count:,} pixels ({percentage:.1f}%)")
                
                all_values.extend(data.flatten())
                file_stats.append({
                    'file': cls_file.name,
                    'unique_values': unique_vals,
                    'counts': counts,
                    'total_pixels': total_pixels
                })
                
        except Exception as e:
            print(f"Error reading {cls_file.name}: {e}")
    
    # Overall statistics
    print(f"\n📊 OVERALL STATISTICS (from {len(sample_files)} files)")
    print("=" * 50)
    
    overall_counter = Counter(all_values)
    total_pixels = len(all_values)
    
    print(f"Total pixels analyzed: {total_pixels:,}")
    print(f"Classes found across all files:")
    
    for class_val, count in sorted(overall_counter.items()):
        percentage = (count / total_pixels) * 100
        print(f"  Class {class_val}: {count:,} pixels ({percentage:.2f}%)")
    
    # Visualize class distribution
    plt.figure(figsize=(12, 6))
    
    classes = sorted(overall_counter.keys())
    counts = [overall_counter[c] for c in classes]
    percentages = [(c/total_pixels)*100 for c in counts]
    
    plt.subplot(1, 2, 1)
    plt.bar(classes, counts)
    plt.xlabel('Class Value')
    plt.ylabel('Pixel Count')
    plt.title('Class Distribution (Raw Counts)')
    plt.yscale('log')
    
    plt.subplot(1, 2, 2)
    plt.bar(classes, percentages)
    plt.xlabel('Class Value')
    plt.ylabel('Percentage')
    plt.title('Class Distribution (Percentages)')
    
    plt.tight_layout()
    plt.savefig('cls_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Recommend building classes
    print(f"\n🏢 BUILDING CLASS RECOMMENDATIONS:")
    print("=" * 50)
    
    # Common SpaceNet class mappings (these are typical values)
    spacenet_classes = {
        1: "Background/No Building",
        2: "Background/No Building", 
        3: "Background/No Building",
        4: "Background/No Building",
        5: "Building",
        6: "Building", 
        7: "Building",
        8: "Building",
        9: "Building",
        10: "Building",
        11: "Building",
        65: "Building/Structure"  # Often used for buildings in SpaceNet
    }
    
    print("Based on typical SpaceNet conventions:")
    for class_val in sorted(overall_counter.keys()):
        if class_val in spacenet_classes:
            print(f"  Class {class_val}: {spacenet_classes[class_val]}")
        else:
            if class_val >= 5:
                print(f"  Class {class_val}: Likely Building")
            else:
                print(f"  Class {class_val}: Likely Background")
    
    # Most common approach for SpaceNet
    print(f"\n🎯 RECOMMENDED APPROACH:")
    print("For SpaceNet datasets, typically:")
    print("  Background: Classes 1-4 (including 2)")
    print("  Buildings: Classes 5+ (including 65)")
    print("  Building threshold: Use values >= 5 as buildings")
    
    return overall_counter, file_stats

if __name__ == "__main__":
    truth_dir = "/Users/umutmurat/Documents/Data/Track1-Truth 2"
    overall_counter, file_stats = analyze_cls_values(truth_dir, sample_size=20)