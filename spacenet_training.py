#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 24 01:17:08 2025

@author: umutmurat
"""

"""
Training script for SpaceNet building segmentation
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
    
class SpaceNetDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transform=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image and mask
        image = cv2.imread(self.image_paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)
        mask = mask.astype(np.float32) / 255.0  # Normalize to 0-1
        
        # Apply transforms
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']
        
        return image, mask.unsqueeze(0)  # Add channel dimension to mask

def get_training_transforms():
    return A.Compose([
        A.OneOf([
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10),
            A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1),
        ], p=0.5),
        A.OneOf([
            A.Blur(blur_limit=3),
            A.GaussNoise(var_limit=(10, 25)),
        ], p=0.2),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

def get_validation_transforms():
    return A.Compose([
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

class CombinedLoss(nn.Module):
    def __init__(self, focal_weight=0.7, dice_weight=0.3):
        super().__init__()
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
        
    def focal_loss(self, inputs, targets, alpha=0.8, gamma=2):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = alpha * (1-pt)**gamma * bce_loss
        return focal_loss.mean()
    
    def dice_loss(self, inputs, targets, smooth=1):
        inputs = torch.sigmoid(inputs)
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        intersection = (inputs * targets).sum()
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)
        return 1 - dice
        
    def forward(self, inputs, targets):
        focal = self.focal_loss(inputs, targets)
        dice = self.dice_loss(inputs, targets)
        return self.focal_weight * focal + self.dice_weight * dice

def calculate_metrics(pred, target, threshold=0.5):
    """Calculate IoU, Precision, Recall, and F1 Score"""
    pred_binary = (torch.sigmoid(pred) > threshold).float()
    target_binary = target
    
    pred_flat = pred_binary.view(-1)
    target_flat = target_binary.view(-1)
    
    intersection = (pred_flat * target_flat).sum()
    union = pred_flat.sum() + target_flat.sum() - intersection
    
    iou = intersection / (union + 1e-8)
    
    tp = intersection
    fp = pred_flat.sum() - tp
    fn = target_flat.sum() - tp
    
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    
    return {
        'iou': iou.item(),
        'precision': precision.item(),
        'recall': recall.item(),
        'f1': f1.item()
    }

def train_epoch(model, dataloader, criterion, optimizer, device, epoch):
    model.train()
    running_loss = 0.0
    metrics = {'iou': 0, 'precision': 0, 'recall': 0, 'f1': 0}
    
    pbar = tqdm(dataloader, desc=f'Training Epoch {epoch}')
    for batch_idx, (images, masks) in enumerate(pbar):
        images, masks = images.to(device), masks.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        # Calculate metrics
        batch_metrics = calculate_metrics(outputs, masks)
        for key in metrics:
            metrics[key] += batch_metrics[key]
        
        # Update progress bar
        pbar.set_postfix({
            'loss': running_loss / (batch_idx + 1),
            'iou': metrics['iou'] / (batch_idx + 1)
        })
    
    for key in metrics:
        metrics[key] /= len(dataloader)
    
    return running_loss / len(dataloader), metrics

def validate_epoch(model, dataloader, criterion, device, epoch):
    model.eval()
    running_loss = 0.0
    metrics = {'iou': 0, 'precision': 0, 'recall': 0, 'f1': 0}
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc=f'Validation Epoch {epoch}')
        for batch_idx, (images, masks) in enumerate(pbar):
            images, masks = images.to(device), masks.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            running_loss += loss.item()
            
            batch_metrics = calculate_metrics(outputs, masks)
            for key in metrics:
                metrics[key] += batch_metrics[key]
            
            pbar.set_postfix({
                'loss': running_loss / (batch_idx + 1),
                'iou': metrics['iou'] / (batch_idx + 1)
            })
    
    for key in metrics:
        metrics[key] /= len(dataloader)
    
    return running_loss / len(dataloader), metrics

def load_paths(file_path):
    """Load file paths from text file"""
    with open(file_path, 'r') as f:
        return [line.strip() for line in f if line.strip()]

def visualize_predictions(model, dataloader, device, save_path, num_samples=6):
    """Visualize model predictions"""
    model.eval()
    
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4*num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(dataloader):
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            predictions = torch.sigmoid(outputs) > 0.5
            
            for i in range(min(num_samples, images.size(0))):
                # Convert tensors to numpy
                img = images[i].cpu().permute(1, 2, 0).numpy()
                # Denormalize
                img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
                img = np.clip(img, 0, 1)
                
                mask = masks[i, 0].cpu().numpy()
                pred = predictions[i, 0].cpu().numpy()
                
                # Plot
                axes[i, 0].imshow(img)
                axes[i, 0].set_title('Input Image')
                axes[i, 0].axis('off')
                
                axes[i, 1].imshow(mask, cmap='gray')
                axes[i, 1].set_title('Ground Truth')
                axes[i, 1].axis('off')
                
                axes[i, 2].imshow(pred, cmap='gray')
                axes[i, 2].set_title('Prediction')
                axes[i, 2].axis('off')
            
            break
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='SpaceNet Building Segmentation Training')
    parser.add_argument('--data_dir', default='spacenet_building_dataset', help='Dataset directory')
    parser.add_argument('--model', default='unetplusplus', choices=['unet', 'unetplusplus', 'deeplabv3plus'])
    parser.add_argument('--encoder', default='resnet50', help='Encoder backbone')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')
    parser.add_argument('--save_dir', default='./checkpoints')
    parser.add_argument('--use_wandb', action='store_true', help='Use Weights & Biases logging')
    parser.add_argument('--project_name', default='spacenet-buildings', help='W&B project name')
    
    args = parser.parse_args()
    
    # Initialize wandb if requested
    if args.use_wandb:
        wandb.init(project=args.project_name, config=args)
    
    # Device
    # Use MPS for Apple Silicon
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print('Using Apple Metal Performance Shaders (MPS)')
    elif torch.cuda.is_available():
        device = torch.device("cuda") 
        print('Using CUDA')
    else:
        device = torch.device("cpu")
        print('Using CPU')
        print(f'Using device: {device}')
    
    # Load data paths
    data_dir = Path(args.data_dir)
    train_images = load_paths(data_dir / 'train_images.txt')
    train_masks = load_paths(data_dir / 'train_masks.txt')
    val_images = load_paths(data_dir / 'val_images.txt')
    val_masks = load_paths(data_dir / 'val_masks.txt')
    
    print(f'Training samples: {len(train_images)}')
    print(f'Validation samples: {len(val_images)}')
    
    # Datasets and dataloaders
    train_dataset = SpaceNetDataset(train_images, train_masks, get_training_transforms())
    val_dataset = SpaceNetDataset(val_images, val_masks, get_validation_transforms())
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, 
                         num_workers=4, pin_memory=False)  # Changed to False
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, 
                       num_workers=4, pin_memory=False)   # Changed to False
    
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
        classes=1,
        activation=None
    )
    model.to(device)
    
    # Loss, optimizer, scheduler
    criterion = CombinedLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Training setup
    save_dir = Path(args.save_dir)
    save_dir.mkdir(exist_ok=True)
    
    best_iou = 0
    patience_counter = 0
    
    print(f"\n🚀 Starting training...")
    print(f"Model: {args.model} with {args.encoder} encoder")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Epochs: {args.epochs}")
    
    # Training loop
    for epoch in range(args.epochs):
        print(f'\nEpoch {epoch + 1}/{args.epochs}')
        print('-' * 30)
        
        # Training
        train_loss, train_metrics = train_epoch(model, train_loader, criterion, optimizer, device, epoch + 1)
        
        # Validation
        val_loss, val_metrics = validate_epoch(model, val_loader, criterion, device, epoch + 1)
        
        # Step scheduler
        scheduler.step()
        
        # Log metrics
        if args.use_wandb:
            wandb.log({
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'train_iou': train_metrics['iou'],
                'val_iou': val_metrics['iou'],
                'train_f1': train_metrics['f1'],
                'val_f1': val_metrics['f1'],
                'lr': optimizer.param_groups[0]['lr']
            })
        
        print(f'Train Loss: {train_loss:.4f}, Train IoU: {train_metrics["iou"]:.4f}')
        print(f'Val Loss: {val_loss:.4f}, Val IoU: {val_metrics["iou"]:.4f}')
        print(f'Val F1: {val_metrics["f1"]:.4f}, Val Precision: {val_metrics["precision"]:.4f}')
        
        # Save best model
        if val_metrics['iou'] > best_iou:
            best_iou = val_metrics['iou']
            patience_counter = 0
            
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_iou': best_iou,
                'val_metrics': val_metrics,
                'config': args
            }, save_dir / 'best_model.pth')
            
            print(f'✅ New best model saved with IoU: {best_iou:.4f}')
            
            # Create visualization
            visualize_predictions(model, val_loader, device, 
                                save_dir / f'predictions_epoch_{epoch+1}.png')
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= args.patience:
            print(f'Early stopping after {args.patience} epochs without improvement')
            break
    
    print(f'\n🎉 Training completed!')
    print(f'Best IoU: {best_iou:.4f}')
    print(f'Model saved at: {save_dir / "best_model.pth"}')
    
    if args.use_wandb:
        wandb.finish()

if __name__ == '__main__':
    main()