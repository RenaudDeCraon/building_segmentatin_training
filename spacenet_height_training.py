#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Building Height Estimation Training Script
Modified from binary segmentation to height regression
FIXED VERSION - Stability and Performance Enhancements
"""
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np
from pathlib import Path
import json
import wandb
from tqdm import tqdm
import matplotlib.pyplot as plt

def get_device():
    """Get the best available device for training"""
    if torch.backends.mps.is_available():
        print("Using Apple Metal Performance Shaders (MPS)")
        return torch.device("mps")
    elif torch.cuda.is_available():
        print("Using CUDA")
        return torch.device("cuda")
    else:
        print("Using CPU")
        return torch.device("cpu")

class HeightEstimationDataset(Dataset):
    """Dataset for building height estimation"""
    def __init__(self, data_list, height_stats, transform=None, normalize_heights=True):
        self.data_list = data_list
        self.transform = transform
        self.normalize_heights = normalize_heights
        
        # Load height statistics for normalization
        self.height_max = height_stats['max']
        self.height_min = height_stats['min']
        # --- MODIFICATION: Use Mean and Std for Z-score normalization ---
        self.height_mean = height_stats['mean']
        self.height_std = height_stats['std']
        
        # Add a tiny epsilon to std to avoid division by zero, though stats should prevent this
        self.height_std_safe = self.height_std + 1e-8 
        
        print(f"Dataset loaded with {len(self.data_list)} samples")
        print(f"Height range: {self.height_min:.2f}m - {self.height_max:.2f}m")
        print(f"Height Mean/Std: {self.height_mean:.2f} / {self.height_std:.2f}") # Added printout
    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        sample = self.data_list[idx]
        
        # Load image
        image = cv2.imread(sample['image'])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load building mask
        mask = cv2.imread(sample['mask'], cv2.IMREAD_GRAYSCALE)
        mask = mask.astype(np.float32) / 255.0
        
        # Load height map
        height_map = np.load(sample['height'])
        
        # Normalize heights
        if self.normalize_heights:
            # --- MODIFICATION: Switched to Z-score normalization for stability ---
            # Z-score normalization: height_norm = (height - mean) / std
            height_map = (height_map - self.height_mean) / self.height_std_safe
        
        # Apply transforms
        if self.transform:
            # Albumentations requires mask to be single channel
            transformed = self.transform(
                image=image, 
                masks=[mask, height_map]
            )
            image = transformed['image']
            mask = transformed['masks'][0]
            height_map = transformed['masks'][1]
        
        # Convert to tensors and add channel dimension
        mask = torch.from_numpy(mask).unsqueeze(0) if isinstance(mask, np.ndarray) else mask.unsqueeze(0)
        height_map = torch.from_numpy(height_map).unsqueeze(0) if isinstance(height_map, np.ndarray) else height_map.unsqueeze(0)
        
        return image, height_map, mask

def get_training_transforms():
    """Training augmentations"""
    return A.Compose([
        # Geometric transforms
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        
        # Color transforms
        A.OneOf([
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2),
        ], p=0.5),
        
        # Noise and blur - FIXED
        A.OneOf([
            A.Blur(blur_limit=3),
            A.GaussNoise(var_limit=(10.0, 50.0)),
        ], p=0.2),
        
        # Normalization
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

def get_validation_transforms():
    """Validation transforms (no augmentation)"""
    return A.Compose([
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

class HeightEstimationLoss(nn.Module):
    """
    Combined loss for height estimation
    - MSE: Penalizes large errors
    - MAE: More robust to outliers
    - Masked: Only calculates loss on building pixels
    """
    def __init__(self, mse_weight=0.7, mae_weight=0.3):
        super().__init__()
        self.mse_weight = mse_weight
        self.mae_weight = mae_weight
        self.mse = nn.MSELoss(reduction='none')
        self.mae = nn.L1Loss(reduction='none')
    
    def forward(self, pred_height, target_height, building_mask):
        """
        Args:
            pred_height: Predicted height map [B, 1, H, W]
            target_height: Ground truth height map [B, 1, H, W]
            building_mask: Binary mask indicating building locations [B, 1, H, W]
        """
        # Only calculate loss on building pixels
        mse_loss = self.mse(pred_height, target_height)
        mae_loss = self.mae(pred_height, target_height)
        
        # Apply mask
        # Note: We rely on the 1e-8 for stability against batches with no buildings
        # This is okay for loss, but we will add a check in the training loop
        # to ensure at least one building exists for metric calculation.
        mask_sum = building_mask.sum()
        if mask_sum < 1.0: # Check if there are any buildings in the batch
             # If no buildings, loss is zero (prevents NaN)
             return torch.zeros(1, device=pred_height.device, requires_grad=True).mean() 

        mse_loss = (mse_loss * building_mask).sum() / (mask_sum + 1e-8)
        mae_loss = (mae_loss * building_mask).sum() / (mask_sum + 1e-8)
        
        total_loss = self.mse_weight * mse_loss + self.mae_weight * mae_loss
        
        return total_loss

# Removed MultiTaskLoss as it was not used and simplifies the fix.

def calculate_height_metrics(pred_height, target_height, building_mask, height_mean, height_std):
    """
    Calculate metrics for height estimation
    Returns metrics in meters (denormalized)
    
    --- MODIFICATION: Uses Mean and Std for Z-score denormalization ---
    """
    # Denormalize predictions: H_m = H_norm * std + mean
    pred_height_m = pred_height * height_std + height_mean
    target_height_m = target_height * height_std + height_mean
    
    # Extract building pixels only
    mask_bool = building_mask > 0.5
    pred_buildings = pred_height_m[mask_bool]
    target_buildings = target_height_m[mask_bool]
    
    if len(pred_buildings) == 0:
        return {
            'rmse': 0.0,
            'mae': 0.0,
            'mape': 0.0,
            'r2': 0.0,
            'num_pixels': 0
        }
    
    # RMSE (Root Mean Squared Error) in meters
    rmse = torch.sqrt(torch.mean((pred_buildings - target_buildings)**2))
    
    # MAE (Mean Absolute Error) in meters
    mae = torch.mean(torch.abs(pred_buildings - target_buildings))
    
    # MAPE (Mean Absolute Percentage Error)
    mape = torch.mean(torch.abs((pred_buildings - target_buildings) / (target_buildings + 1e-8))) * 100
    
    # R² Score
    ss_res = torch.sum((target_buildings - pred_buildings)**2)
    # --- MODIFICATION: Use target_buildings.mean() for ss_tot calculation ---
    ss_tot = torch.sum((target_buildings - target_buildings.mean())**2)
    r2 = 1 - ss_res / (ss_tot + 1e-8)
    
    return {
        'rmse': rmse.item(),
        'mae': mae.item(),
        'mape': mape.item(),
        'r2': r2.item(),
        'num_pixels': len(pred_buildings)
    }

# --- MODIFICATION: Added height_mean and height_std to training function arguments ---
def train_epoch(model, dataloader, criterion, optimizer, device, epoch, height_mean, height_std):
    """Training loop for one epoch"""
    model.train()
    running_loss = 0.0
    metrics = {'rmse': 0, 'mae': 0, 'mape': 0, 'r2': 0, 'num_batches': 0}
    
    pbar = tqdm(dataloader, desc=f'Training Epoch {epoch}')
    for batch_idx, (images, target_heights, building_masks) in enumerate(pbar):
        images = images.to(device)
        target_heights = target_heights.to(device)
        building_masks = building_masks.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        pred_heights = model(images)
        
        # Check for non-building batch (to avoid unstable metrics)
        if building_masks.sum() < 1.0:
            pbar.set_postfix({'loss': 'Skip (No Buildings)'})
            continue # Skip batch with no buildings in it
            
        # ADDED: NaN/Inf check
        if torch.isnan(pred_heights).any() or torch.isinf(pred_heights).any():
            print(f"\n⚠️  WARNING: NaN/Inf detected in predictions at batch {batch_idx}. Skipping batch.")
            continue  # Skip this batch
        
        # Calculate loss
        # Loss is robust to non-building batches due to check in HeightEstimationLoss
        loss = criterion(pred_heights, target_heights, building_masks)
        
        # Check loss for NaN
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"\n⚠️  WARNING: NaN/Inf loss at batch {batch_idx}. Skipping batch.")
            continue
        
        # Backward pass
        loss.backward()
        
        # ADDED: Gradient clipping (Keep this)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        running_loss += loss.item()
        
        # Calculate metrics
        with torch.no_grad():
            batch_metrics = calculate_height_metrics(
                pred_heights, target_heights, building_masks, height_mean, height_std
            )
            for key in ['rmse', 'mae', 'mape', 'r2']:
                metrics[key] += batch_metrics[key]
            metrics['num_batches'] += 1
        
        # Update progress bar
        current_mae = metrics['mae'] / metrics['num_batches'] if metrics['num_batches'] > 0 else 0
        current_loss = running_loss / metrics['num_batches'] if metrics['num_batches'] > 0 else 0
        pbar.set_postfix({
            'loss': f'{current_loss:.4f}',
            'mae': f'{current_mae:.2f}m'
        })
    
    # Average metrics
    total_batches = metrics['num_batches']
    if total_batches == 0:
        return 0.0, {'rmse': 0, 'mae': 0, 'mape': 0, 'r2': 0}

    for key in ['rmse', 'mae', 'mape', 'r2']:
        metrics[key] /= total_batches
    
    return running_loss / total_batches, {k: metrics[k] for k in ['rmse', 'mae', 'mape', 'r2']}

# --- MODIFICATION: Added height_mean and height_std to validation function arguments ---
def validate_epoch(model, dataloader, criterion, device, epoch, height_mean, height_std):
    """Validation loop for one epoch"""
    model.eval()
    running_loss = 0.0
    metrics = {'rmse': 0, 'mae': 0, 'mape': 0, 'r2': 0, 'num_batches': 0}
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc=f'Validation Epoch {epoch}')
        for batch_idx, (images, target_heights, building_masks) in enumerate(pbar):
            images = images.to(device)
            target_heights = target_heights.to(device)
            building_masks = building_masks.to(device)
            
            # Forward pass
            pred_heights = model(images)
            
            # Check for non-building batch
            if building_masks.sum() < 1.0:
                 pbar.set_postfix({'loss': 'Skip (No Buildings)'})
                 continue
            
            # --- MODIFICATION: Check for NaN in Validation Predictions ---
            if torch.isnan(pred_heights).any() or torch.isinf(pred_heights).any():
                print(f"\n⚠️  WARNING: NaN/Inf detected in validation predictions at batch {batch_idx}. Skipping batch.")
                continue

            # Calculate loss
            loss = criterion(pred_heights, target_heights, building_masks)
            
            # --- MODIFICATION: Check for NaN in Validation Loss ---
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"\n⚠️  WARNING: NaN/Inf loss in validation at batch {batch_idx}. Skipping batch.")
                continue

            running_loss += loss.item()
            
            # Calculate metrics
            batch_metrics = calculate_height_metrics(
                pred_heights, target_heights, building_masks, height_mean, height_std
            )
            for key in ['rmse', 'mae', 'mape', 'r2']:
                metrics[key] += batch_metrics[key]
            metrics['num_batches'] += 1
            
            current_mae = metrics['mae'] / metrics['num_batches'] if metrics['num_batches'] > 0 else 0
            current_loss = running_loss / metrics['num_batches'] if metrics['num_batches'] > 0 else 0
            pbar.set_postfix({
                'loss': f'{current_loss:.4f}',
                'mae': f'{current_mae:.2f}m'
            })
    
    # Average metrics
    total_batches = metrics['num_batches']
    if total_batches == 0:
        # If no batches were processed (e.g., all were non-building), return 0.0
        return 0.0, {'rmse': 0, 'mae': 0, 'mape': 0, 'r2': 0}

    for key in ['rmse', 'mae', 'mape', 'r2']:
        metrics[key] /= total_batches
    
    return running_loss / total_batches, {k: metrics[k] for k in ['rmse', 'mae', 'mape', 'r2']}

# --- MODIFICATION: Added height_mean and height_std to visualization arguments ---
def visualize_predictions(model, dataloader, device, height_mean, height_std, save_path, num_samples=4):
    """Visualize model predictions"""
    model.eval()
    
    fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4*num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    # --- MODIFICATION: Filter out non-building batches for visualization ---
    valid_batches = []
    for batch in dataloader:
        if batch[2].sum() > 0: # Check if building mask has pixels
            valid_batches.append(batch)
            if len(valid_batches) >= 1: break # Only need one valid batch

    if not valid_batches:
        print("Skipping visualization: No batches with buildings found in dataloader.")
        return

    images_batch, target_heights_batch, masks_batch = valid_batches[0]

    with torch.no_grad():
        images_batch = images_batch.to(device)
        target_heights_batch = target_heights_batch.to(device)
        masks_batch = masks_batch.to(device)
        
        pred_heights_batch = model(images_batch)
        
        for i in range(min(num_samples, images_batch.size(0))):
            # Convert to numpy and denormalize
            img = images_batch[i].cpu().permute(1, 2, 0).numpy()
            img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
            img = np.clip(img, 0, 1)
            
            mask = masks_batch[i, 0].cpu().numpy()
            
            # --- MODIFICATION: Denormalize using Z-score stats ---
            target_height = target_heights_batch[i, 0].cpu().numpy() * height_std + height_mean
            pred_height = pred_heights_batch[i, 0].cpu().numpy() * height_std + height_mean
            
            # Clip denormalized height to min 0m for visualization, as negative height is non-physical
            pred_height = np.clip(pred_height, 0, target_height.max() * 2) 
            target_height = np.clip(target_height, 0, target_height.max() * 2)
            
            # Use max height from stats for consistent color bar range
            v_max = max(height_mean + 2 * height_std, target_height.max()) 

            # Calculate error map
            error_map = np.abs(target_height - pred_height) * mask
            
            # Plot
            axes[i, 0].imshow(img)
            axes[i, 0].set_title('Input Image')
            axes[i, 0].axis('off')
            
            im1 = axes[i, 1].imshow(target_height, cmap='viridis', vmin=0, vmax=v_max)
            axes[i, 1].set_title(f'Ground Truth\nMax: {target_height[mask>0].max():.1f}m')
            axes[i, 1].axis('off')
            plt.colorbar(im1, ax=axes[i, 1], fraction=0.046)
            
            im2 = axes[i, 2].imshow(pred_height, cmap='viridis', vmin=0, vmax=v_max)
            axes[i, 2].set_title(f'Prediction\nMax: {pred_height[mask>0].max():.1f}m')
            axes[i, 2].axis('off')
            plt.colorbar(im2, ax=axes[i, 2], fraction=0.046)
            
            im3 = axes[i, 3].imshow(error_map, cmap='hot', vmin=0, vmax=10)
            axes[i, 3].set_title(f'Error Map\nMAE: {error_map[mask>0].mean():.2f}m')
            axes[i, 3].axis('off')
            plt.colorbar(im3, ax=axes[i, 3], fraction=0.046)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Building Height Estimation Training')
    parser.add_argument('--data_dir', default='spacenet_height_dataset', help='Dataset directory')
    parser.add_argument('--model', default='unetplusplus', choices=['unet', 'unetplusplus', 'deeplabv3plus'])
    parser.add_argument('--encoder', default='resnet50', help='Encoder backbone')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')
    parser.add_argument('--save_dir', default='./height_checkpoints')
    parser.add_argument('--use_wandb', action='store_true', help='Use Weights & Biases logging')
    parser.add_argument('--project_name', default='spacenet-height-estimation', help='W&B project name')
    
    args = parser.parse_args()
    
    # Initialize wandb if requested
    if args.use_wandb:
        wandb.init(project=args.project_name, config=args)
    
    # Device
    device = get_device()
    print(f'Using device: {device}')
    
    # Load data paths and statistics
    data_dir = Path(args.data_dir)
    
    # Load height statistics
    try:
        with open(data_dir / 'height_statistics.json', 'r') as f:
            height_stats = json.load(f)
    except FileNotFoundError:
        print(f"ERROR: height_statistics.json not found in {data_dir}. Cannot proceed.")
        return # Exit if critical file is missing
        
    print(f"\n📊 Height Statistics:")
    print(f"  Min: {height_stats['min']:.2f}m")
    print(f"  Max: {height_stats['max']:.2f}m")
    print(f"  Mean: {height_stats['mean']:.2f}m")
    print(f"  Std: {height_stats['std']:.2f}m") # Added printout
    
    # Extract mean and std for metric calculation
    height_mean = height_stats['mean']
    height_std = height_stats['std']
    
    # Load data splits
    with open(data_dir / 'train_data.json', 'r') as f:
        train_data = json.load(f)
    
    with open(data_dir / 'val_data.json', 'r') as f:
        val_data = json.load(f)
    
    print(f'\n📂 Dataset:')
    print(f'  Training samples: {len(train_data)}')
    print(f'  Validation samples: {len(val_data)}')
    
    # Create datasets
    train_dataset = HeightEstimationDataset(
        train_data, height_stats, get_training_transforms()
    )
    val_dataset = HeightEstimationDataset(
        val_data, height_stats, get_validation_transforms()
    )
    
    # Create dataloaders - FIXED: num_workers=0 for Mac stability
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=0,  # CHANGED from 4 to 0
        pin_memory=False
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=0,  # CHANGED from 4 to 0
        pin_memory=False
    )
    
    # Model
    model_map = {
        'unet': smp.Unet,
        'unetplusplus': smp.UnetPlusPlus,
        'deeplabv3plus': smp.DeepLabV3Plus,
    }
    
    model = model_map[args.model](
        encoder_name=args.encoder,
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,  # Single channel for height
        activation=None  # No activation for regression
    )
    model.to(device)
    
    # Loss, optimizer, scheduler
    criterion = HeightEstimationLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=args.lr, 
        weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=args.epochs
    )
    
    # Training setup
    save_dir = Path(args.save_dir)
    save_dir.mkdir(exist_ok=True)
    
    best_mae = float('inf')
    patience_counter = 0
    
    print(f"\n🚀 Starting training...")
    print(f"  Model: {args.model} with {args.encoder} encoder")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Epochs: {args.epochs}")
    
    # Training loop
    for epoch in range(args.epochs):
        print(f'\n{"="*60}')
        print(f'Epoch {epoch + 1}/{args.epochs}')
        print(f'{"="*60}')
        
        # Training
        train_loss, train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, device, 
            epoch + 1, height_mean, height_std # Pass mean/std
        )
        
        # Validation
        val_loss, val_metrics = validate_epoch(
            model, val_loader, criterion, device, 
            epoch + 1, height_mean, height_std # Pass mean/std
        )
        
        # Step scheduler
        scheduler.step()
        
        # Log metrics
        if args.use_wandb:
            wandb.log({
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'train_mae': train_metrics['mae'],
                'val_mae': val_metrics['mae'],
                'train_rmse': train_metrics['rmse'],
                'val_rmse': val_metrics['rmse'],
                'val_r2': val_metrics['r2'],
                'lr': optimizer.param_groups[0]['lr']
            })
        
        print(f'\n📈 Training Results:')
        print(f'  Loss: {train_loss:.4f}')
        print(f'  MAE: {train_metrics["mae"]:.2f}m')
        print(f'  RMSE: {train_metrics["rmse"]:.2f}m')
        print(f'  R²: {train_metrics["r2"]:.4f}')
        
        print(f'\n📉 Validation Results:')
        print(f'  Loss: {val_loss:.4f}')
        print(f'  MAE: {val_metrics["mae"]:.2f}m')
        print(f'  RMSE: {val_metrics["rmse"]:.2f}m')
        print(f'  R²: {val_metrics["r2"]:.4f}')
        
        # Save best model
        if val_metrics['mae'] < best_mae and val_metrics['mae'] != 0.0:
            best_mae = val_metrics['mae']
            patience_counter = 0
            
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_mae': best_mae,
                'val_metrics': val_metrics,
                'height_stats': height_stats,
                'config': args
            }, save_dir / 'best_model.pth')
            
            print(f'\n✅ New best model saved with MAE: {best_mae:.2f}m')
            
            # Create visualization
            visualize_predictions(
                model, val_loader, device, height_mean, height_std,
                save_dir / f'predictions_epoch_{epoch+1}.png'
            )
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= args.patience:
            print(f'\n⏹️  Early stopping after {args.patience} epochs without improvement')
            break
    
    print(f'\n{"="*60}')
    print(f'🎉 Training completed!')
    print(f'{"="*60}')
    print(f'Best MAE: {best_mae:.2f}m')
    print(f'Model saved at: {save_dir / "best_model.pth"}')
    
    if args.use_wandb:
        wandb.finish()

if __name__ == '__main__':
    main()