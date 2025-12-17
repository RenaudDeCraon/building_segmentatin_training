#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Building Height Estimation GUI - Enhanced with Analytics
Load a trained model checkpoint and predict heights from satellite images
"""
import tkinter as tk
from tkinter import filedialog, ttk, messagebox, scrolledtext
from PIL import Image, ImageTk
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
import numpy as np
import cv2
import json
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib
matplotlib.use('TkAgg')
from datetime import datetime

class HeightEstimationGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Building Height Estimation - Advanced Analytics")
        self.root.geometry("1600x1000")
        
        # Variables
        self.checkpoint_path = None
        self.image_path = None
        self.model = None
        self.device = None
        self.height_stats = None
        self.config = None
        self.current_prediction = None
        self.current_image = None
        
        # Setup UI
        self.setup_ui()
        
    def setup_ui(self):
        """Setup the GUI layout"""
        # Main container with notebook (tabs)
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create tabs
        self.main_tab = ttk.Frame(self.notebook)
        self.analytics_tab = ttk.Frame(self.notebook)
        self.hyperparams_tab = ttk.Frame(self.notebook)
        
        self.notebook.add(self.main_tab, text="  Prediction  ")
        self.notebook.add(self.analytics_tab, text="  Analytics  ")
        self.notebook.add(self.hyperparams_tab, text="  Model Info  ")
        
        # Setup each tab
        self.setup_main_tab()
        self.setup_analytics_tab()
        self.setup_hyperparams_tab()
        
        # Status bar (common)
        self.status_var = tk.StringVar(value="Ready. Please load a checkpoint and image.")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, 
                              anchor=tk.W, font=('Arial', 9))
        status_bar.pack(side=tk.BOTTOM, fill=tk.X, padx=5, pady=2)
    
    def setup_main_tab(self):
        """Setup main prediction tab"""
        main_frame = ttk.Frame(self.main_tab, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Configure grid
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(2, weight=1)
        
        # ===== File Selection =====
        file_frame = ttk.LabelFrame(main_frame, text="File Selection", padding="10")
        file_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        file_frame.columnconfigure(1, weight=1)
        
        ttk.Label(file_frame, text="Model Checkpoint:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.checkpoint_label = ttk.Label(file_frame, text="No checkpoint loaded", foreground="gray")
        self.checkpoint_label.grid(row=0, column=1, sticky=tk.W, padx=10)
        ttk.Button(file_frame, text="Browse", command=self.load_checkpoint).grid(row=0, column=2, padx=5)
        
        ttk.Label(file_frame, text="Input Image:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.image_label = ttk.Label(file_frame, text="No image selected", foreground="gray")
        self.image_label.grid(row=1, column=1, sticky=tk.W, padx=10)
        ttk.Button(file_frame, text="Browse", command=self.load_image).grid(row=1, column=2, padx=5)
        
        # Buttons
        button_frame = ttk.Frame(file_frame)
        button_frame.grid(row=2, column=0, columnspan=3, pady=10)
        
        self.predict_button = ttk.Button(button_frame, text="🚀 Estimate Heights", 
                                        command=self.predict_height, state=tk.DISABLED)
        self.predict_button.grid(row=0, column=0, padx=5)
        
        self.export_button = ttk.Button(button_frame, text="💾 Export Results", 
                                       command=self.export_results, state=tk.DISABLED)
        self.export_button.grid(row=0, column=1, padx=5)
        
        ttk.Button(button_frame, text="🗑️ Clear", command=self.clear_results).grid(row=0, column=2, padx=5)
        
        # ===== Quick Stats =====
        stats_frame = ttk.LabelFrame(main_frame, text="Quick Statistics", padding="10")
        stats_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        stats_frame.columnconfigure((0,1,2,3), weight=1)
        
        # Create stat labels
        self.stat_labels = {}
        stats = [
            ("Max Height", "max_height"),
            ("Mean Height", "mean_height"),
            ("Median Height", "median_height"),
            ("Building Coverage", "coverage")
        ]
        
        for i, (label, key) in enumerate(stats):
            frame = ttk.Frame(stats_frame)
            frame.grid(row=0, column=i, padx=10, pady=5)
            
            ttk.Label(frame, text=label, font=('Arial', 9)).pack()
            self.stat_labels[key] = ttk.Label(frame, text="--", 
                                             font=('Arial', 16, 'bold'), foreground='blue')
            self.stat_labels[key].pack()
        
        # ===== Visualization =====
        viz_frame = ttk.LabelFrame(main_frame, text="Prediction Results", padding="10")
        viz_frame.grid(row=2, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        self.fig_main, self.axes_main = plt.subplots(1, 3, figsize=(15, 4.5))
        self.canvas_main = FigureCanvasTkAgg(self.fig_main, master=viz_frame)
        self.canvas_main.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        for ax in self.axes_main:
            ax.axis('off')
        self.fig_main.patch.set_facecolor('#f0f0f0')
        self.canvas_main.draw()
    
    def setup_analytics_tab(self):
        """Setup analytics tab with detailed statistics"""
        analytics_frame = ttk.Frame(self.analytics_tab, padding="10")
        analytics_frame.pack(fill=tk.BOTH, expand=True)
        
        # Left: Detailed plots
        plot_frame = ttk.LabelFrame(analytics_frame, text="Detailed Analysis", padding="10")
        plot_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        self.fig_analytics, self.axes_analytics = plt.subplots(2, 2, figsize=(10, 8))
        self.canvas_analytics = FigureCanvasTkAgg(self.fig_analytics, master=plot_frame)
        self.canvas_analytics.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Right: Statistics text
        text_frame = ttk.LabelFrame(analytics_frame, text="Detailed Statistics", padding="10")
        text_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))
        
        self.analytics_text = scrolledtext.ScrolledText(text_frame, width=50, height=40,
                                                        font=('Courier', 10))
        self.analytics_text.pack(fill=tk.BOTH, expand=True)
        
        # Export analytics button
        ttk.Button(text_frame, text="📊 Export Analytics", 
                  command=self.export_analytics).pack(pady=5)
    
    def setup_hyperparams_tab(self):
        """Setup hyperparameters and model info tab"""
        hyper_frame = ttk.Frame(self.hyperparams_tab, padding="10")
        hyper_frame.pack(fill=tk.BOTH, expand=True)
        
        # Left: Model architecture
        arch_frame = ttk.LabelFrame(hyper_frame, text="Model Architecture", padding="10")
        arch_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        self.arch_text = scrolledtext.ScrolledText(arch_frame, width=60, height=40,
                                                   font=('Courier', 10))
        self.arch_text.pack(fill=tk.BOTH, expand=True)
        
        # Right: Training hyperparameters
        hyper_right = ttk.Frame(hyper_frame)
        hyper_right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))
        
        params_frame = ttk.LabelFrame(hyper_right, text="Training Hyperparameters", padding="10")
        params_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 5))
        
        self.params_text = scrolledtext.ScrolledText(params_frame, width=60, height=20,
                                                     font=('Courier', 10))
        self.params_text.pack(fill=tk.BOTH, expand=True)
        
        # Normalization info
        norm_frame = ttk.LabelFrame(hyper_right, text="Data Normalization", padding="10")
        norm_frame.pack(fill=tk.BOTH, expand=True, pady=(5, 0))
        
        self.norm_text = scrolledtext.ScrolledText(norm_frame, width=60, height=15,
                                                   font=('Courier', 10))
        self.norm_text.pack(fill=tk.BOTH, expand=True)
    
    def load_checkpoint(self):
        """Load model checkpoint"""
        filepath = filedialog.askopenfilename(
            title="Select Model Checkpoint",
            filetypes=[("PyTorch Checkpoint", "*.pth"), ("All Files", "*.*")]
        )
        
        if not filepath:
            return
        
        try:
            self.status_var.set("Loading checkpoint...")
            self.root.update()
            
            # Determine device
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
                device_name = "Apple MPS"
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
                device_name = f"CUDA ({torch.cuda.get_device_name(0)})"
            else:
                self.device = torch.device("cpu")
                device_name = "CPU"
            
            # Load checkpoint with weights_only=False
            checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
            
            # Extract info
            self.config = checkpoint.get('config')
            self.height_stats = checkpoint.get('height_stats')
            val_metrics = checkpoint.get('val_metrics', {})
            epoch = checkpoint.get('epoch', 'Unknown')
            
            if self.height_stats is None or self.config is None:
                raise ValueError("Checkpoint missing required information")
            
            # Create model
            model_type = self.config.model if hasattr(self.config, 'model') else 'unetplusplus'
            encoder_name = self.config.encoder if hasattr(self.config, 'encoder') else 'resnet50'
            
            model_map = {
                'unet': smp.Unet,
                'unetplusplus': smp.UnetPlusPlus,
                'deeplabv3plus': smp.DeepLabV3Plus,
            }
            
            self.model = model_map[model_type](
                encoder_name=encoder_name,
                encoder_weights=None,
                in_channels=3,
                classes=1,
                activation=None
            )
            
            # Load weights
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)
            self.model.eval()
            
            self.checkpoint_path = filepath
            self.checkpoint_label.config(text=Path(filepath).name, foreground="green")
            
            # Update all info displays
            self.update_hyperparameters_display(device_name, val_metrics, epoch)
            
            if self.image_path:
                self.predict_button.config(state=tk.NORMAL)
            
            self.status_var.set(f"✅ Checkpoint loaded: {Path(filepath).name} on {device_name}")
            messagebox.showinfo("Success", f"Model loaded!\nDevice: {device_name}\nEpoch: {epoch}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load checkpoint:\n{str(e)}")
            self.status_var.set("Error loading checkpoint")
    
    def update_hyperparameters_display(self, device_name, val_metrics, epoch):
        """Update hyperparameters tab with model info"""
        # Model Architecture
        self.arch_text.delete(1.0, tk.END)
        arch_info = f"""
{'='*60}
MODEL ARCHITECTURE
{'='*60}

Model Type:        {self.config.model if hasattr(self.config, 'model') else 'Unknown'}
Encoder:           {self.config.encoder if hasattr(self.config, 'encoder') else 'Unknown'}
Input Channels:    3 (RGB)
Output Channels:   1 (Height Map)
Input Size:        256x256
Activation:        None (Regression)

Device:            {device_name}
Training Epoch:    {epoch}

{'='*60}
MODEL PERFORMANCE (Validation Set)
{'='*60}

MAE (Mean Absolute Error):    {val_metrics.get('mae', 'N/A'):.2f} meters
RMSE (Root Mean Squared):     {val_metrics.get('rmse', 'N/A'):.2f} meters
R² Score:                     {val_metrics.get('r2', 'N/A'):.4f}
MAPE:                         {val_metrics.get('mape', 'N/A'):.2f}%

{'='*60}
MODEL PARAMETERS
{'='*60}

Total Parameters:  {sum(p.numel() for p in self.model.parameters()):,}
Trainable Params:  {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}
"""
        self.arch_text.insert(1.0, arch_info)
        self.arch_text.config(state=tk.DISABLED)
        
        # Training Hyperparameters
        self.params_text.delete(1.0, tk.END)
        params_info = f"""
{'='*60}
TRAINING HYPERPARAMETERS
{'='*60}

Epochs:            {self.config.epochs if hasattr(self.config, 'epochs') else 'N/A'}
Batch Size:        {self.config.batch_size if hasattr(self.config, 'batch_size') else 'N/A'}
Learning Rate:     {self.config.lr if hasattr(self.config, 'lr') else 'N/A'}
Weight Decay:      {self.config.weight_decay if hasattr(self.config, 'weight_decay') else 'N/A'}
Patience:          {self.config.patience if hasattr(self.config, 'patience') else 'N/A'}

Optimizer:         AdamW
Scheduler:         CosineAnnealingLR
Loss Function:     Combined MSE + MAE
  - MSE Weight:    0.7
  - MAE Weight:    0.3

Data Augmentation:
  - Horizontal Flip:     50%
  - Vertical Flip:       50%
  - Random Rotate 90:    50%
  - Color Jitter:        50%
  - Gaussian Noise:      20%
  - Blur:                20%
"""
        self.params_text.insert(1.0, params_info)
        self.params_text.config(state=tk.DISABLED)
        
        # Normalization Info
        self.norm_text.delete(1.0, tk.END)
        norm_info = f"""
{'='*60}
HEIGHT NORMALIZATION STATISTICS
{'='*60}

Method:            Z-Score Normalization
Formula:           (height - mean) / std

Dataset Statistics:
  Minimum:         {self.height_stats['min']:.2f} meters
  Maximum:         {self.height_stats['max']:.2f} meters
  Mean:            {self.height_stats['mean']:.2f} meters
  Std Dev:         {self.height_stats['std']:.2f} meters
  Total Samples:   {self.height_stats.get('samples', 'N/A'):,}

Image Normalization:
  Mean RGB:        [0.485, 0.456, 0.406]
  Std RGB:         [0.229, 0.224, 0.225]
  (ImageNet Statistics)
"""
        self.norm_text.insert(1.0, norm_info)
        self.norm_text.config(state=tk.DISABLED)
    
    def load_image(self):
        """Load input image"""
        filepath = filedialog.askopenfilename(
            title="Select Satellite Image",
            filetypes=[
                ("Image Files", "*.png *.jpg *.jpeg *.tif *.tiff"),
                ("All Files", "*.*")
            ]
        )
        
        if not filepath:
            return
        
        try:
            image = cv2.imread(filepath)
            if image is None:
                raise ValueError("Could not read image file")
            
            self.image_path = filepath
            self.image_label.config(text=Path(filepath).name, foreground="green")
            
            if self.model:
                self.predict_button.config(state=tk.NORMAL)
            
            img_shape = image.shape
            self.status_var.set(f"✅ Image loaded: {Path(filepath).name} ({img_shape[1]}x{img_shape[0]})")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image:\n{str(e)}")
    
    def get_transforms(self):
        """Get preprocessing transforms"""
        return A.Compose([
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
    
    def predict_height(self):
        """Predict building heights"""
        if not self.model or not self.image_path:
            messagebox.showwarning("Warning", "Please load both checkpoint and image!")
            return
        
        try:
            self.status_var.set("🔄 Processing...")
            self.root.update()
            
            # Load and process image
            image = cv2.imread(self.image_path)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            self.current_image = image_rgb
            original_size = image_rgb.shape[:2]
            
            image_resized = cv2.resize(image_rgb, (256, 256), interpolation=cv2.INTER_LINEAR)
            
            transform = self.get_transforms()
            transformed = transform(image=image_resized)
            image_tensor = transformed['image'].unsqueeze(0).to(self.device)
            
            # Predict
            with torch.no_grad():
                pred_normalized = self.model(image_tensor)
            
            # Denormalize
            pred_normalized = pred_normalized.cpu().numpy()[0, 0]
            pred_height = pred_normalized * self.height_stats['std'] + self.height_stats['mean']
            pred_height = np.clip(pred_height, 0, self.height_stats['max'] * 1.2)
            
            # Resize to original
            pred_height_full = cv2.resize(pred_height, (original_size[1], original_size[0]), 
                                         interpolation=cv2.INTER_LINEAR)
            
            self.current_prediction = pred_height_full
            
            # Update all visualizations
            self.update_quick_stats(pred_height_full)
            self.visualize_main_results(image_rgb, pred_height_full)
            self.update_analytics(pred_height_full)
            
            self.export_button.config(state=tk.NORMAL)
            self.status_var.set("✅ Prediction complete!")
            
        except Exception as e:
            messagebox.showerror("Error", f"Prediction failed:\n{str(e)}")
    
    def update_quick_stats(self, height_map):
        """Update quick statistics display"""
        heights_positive = height_map[height_map > 1]
        
        if len(heights_positive) > 0:
            self.stat_labels['max_height'].config(text=f"{height_map.max():.1f}m")
            self.stat_labels['mean_height'].config(text=f"{heights_positive.mean():.1f}m")
            self.stat_labels['median_height'].config(text=f"{np.median(heights_positive):.1f}m")
            coverage = (len(heights_positive) / height_map.size) * 100
            self.stat_labels['coverage'].config(text=f"{coverage:.1f}%")
        else:
            for key in self.stat_labels:
                self.stat_labels[key].config(text="0.0")
    
    def visualize_main_results(self, original_image, height_map):
        """Visualize main prediction results"""
        for ax in self.axes_main:
            ax.clear()
        
        # Original
        self.axes_main[0].imshow(original_image)
        self.axes_main[0].set_title('Input Image', fontsize=13, fontweight='bold', pad=10)
        self.axes_main[0].axis('off')
        
        # Height map
        im1 = self.axes_main[1].imshow(height_map, cmap='plasma', vmin=0, 
                                       vmax=min(self.height_stats['max'], height_map.max() * 1.2))
        self.axes_main[1].set_title(f'Height Map (Max: {height_map.max():.1f}m)', 
                                    fontsize=13, fontweight='bold', pad=10)
        self.axes_main[1].axis('off')
        cbar = plt.colorbar(im1, ax=self.axes_main[1], fraction=0.046)
        cbar.set_label('Height (m)', rotation=270, labelpad=15)
        
        # Distribution
        heights_positive = height_map[height_map > 1].flatten()
        if len(heights_positive) > 0:
            self.axes_main[2].hist(heights_positive, bins=50, color='steelblue', 
                                  alpha=0.7, edgecolor='black')
            self.axes_main[2].axvline(heights_positive.mean(), color='red', 
                                     linestyle='--', linewidth=2, label=f'Mean: {heights_positive.mean():.1f}m')
            self.axes_main[2].set_xlabel('Height (m)', fontweight='bold')
            self.axes_main[2].set_ylabel('Frequency', fontweight='bold')
            self.axes_main[2].set_title('Distribution', fontsize=13, fontweight='bold', pad=10)
            self.axes_main[2].legend()
            self.axes_main[2].grid(True, alpha=0.3)
        
        self.fig_main.tight_layout()
        self.canvas_main.draw()
    
    def update_analytics(self, height_map):
        """Update detailed analytics"""
        for ax in self.axes_analytics.flat:
            ax.clear()
        
        heights_positive = height_map[height_map > 1].flatten()
        
        if len(heights_positive) == 0:
            self.axes_analytics[0, 0].text(0.5, 0.5, 'No buildings detected', 
                                          ha='center', va='center')
            self.fig_analytics.tight_layout()
            self.canvas_analytics.draw()
            return
        
        # 1. Height distribution with KDE
        self.axes_analytics[0, 0].hist(heights_positive, bins=100, density=True, 
                                       alpha=0.6, color='skyblue', edgecolor='black')
        self.axes_analytics[0, 0].set_xlabel('Height (m)')
        self.axes_analytics[0, 0].set_ylabel('Density')
        self.axes_analytics[0, 0].set_title('Height Distribution', fontweight='bold')
        self.axes_analytics[0, 0].grid(True, alpha=0.3)
        
        # 2. Cumulative distribution
        sorted_heights = np.sort(heights_positive)
        cumulative = np.arange(1, len(sorted_heights) + 1) / len(sorted_heights)
        self.axes_analytics[0, 1].plot(sorted_heights, cumulative * 100, linewidth=2)
        self.axes_analytics[0, 1].set_xlabel('Height (m)')
        self.axes_analytics[0, 1].set_ylabel('Cumulative %')
        self.axes_analytics[0, 1].set_title('Cumulative Distribution', fontweight='bold')
        self.axes_analytics[0, 1].grid(True, alpha=0.3)
        
        # 3. Height categories
        categories = ['Low\n(0-5m)', 'Medium\n(5-15m)', 'High\n(15-30m)', 'Very High\n(>30m)']
        counts = [
            ((heights_positive >= 0) & (heights_positive < 5)).sum(),
            ((heights_positive >= 5) & (heights_positive < 15)).sum(),
            ((heights_positive >= 15) & (heights_positive < 30)).sum(),
            (heights_positive >= 30).sum()
        ]
        colors = ['#90EE90', '#FFD700', '#FFA500', '#FF6347']
        self.axes_analytics[1, 0].bar(categories, counts, color=colors, edgecolor='black')
        self.axes_analytics[1, 0].set_ylabel('Count')
        self.axes_analytics[1, 0].set_title('Building Height Categories', fontweight='bold')
        self.axes_analytics[1, 0].grid(True, alpha=0.3, axis='y')
        
        # 4. Box plot
        self.axes_analytics[1, 1].boxplot(heights_positive, vert=True)
        self.axes_analytics[1, 1].set_ylabel('Height (m)')
        self.axes_analytics[1, 1].set_title('Box Plot Analysis', fontweight='bold')
        self.axes_analytics[1, 1].grid(True, alpha=0.3, axis='y')
        
        self.fig_analytics.tight_layout()
        self.canvas_analytics.draw()
        
        # Update statistics text
        self.update_analytics_text(heights_positive, height_map)
    
    def update_analytics_text(self, heights_positive, height_map):
        """Update analytics text with detailed statistics"""
        self.analytics_text.delete(1.0, tk.END)
        
        # Calculate statistics
        stats_text = f"""
{'='*60}
DETAILED ANALYTICS REPORT
{'='*60}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

{'='*60}
BASIC STATISTICS
{'='*60}

Total Pixels:              {height_map.size:,}
Building Pixels:           {len(heights_positive):,}
Background Pixels:         {height_map.size - len(heights_positive):,}
Building Coverage:         {(len(heights_positive)/height_map.size)*100:.2f}%

{'='*60}
HEIGHT STATISTICS (meters)
{'='*60}

Minimum:                   {heights_positive.min():.2f}
Maximum:                   {heights_positive.max():.2f}
Mean:                      {heights_positive.mean():.2f}
Median:                    {np.median(heights_positive):.2f}
Standard Deviation:        {heights_positive.std():.2f}
Variance:                  {heights_positive.var():.2f}

Quartiles:
  Q1 (25%):                {np.percentile(heights_positive, 25):.2f}
  Q2 (50%, Median):        {np.percentile(heights_positive, 50):.2f}
  Q3 (75%):                {np.percentile(heights_positive, 75):.2f}
  IQR:                     {np.percentile(heights_positive, 75) - np.percentile(heights_positive, 25):.2f}

Percentiles:
  10th:                    {np.percentile(heights_positive, 10):.2f}
  90th:                    {np.percentile(heights_positive, 90):.2f}
  95th:                    {np.percentile(heights_positive, 95):.2f}
  99th:                    {np.percentile(heights_positive, 99):.2f}

{'='*60}
HEIGHT CATEGORIES
{'='*60}

Low Buildings (0-5m):      {((heights_positive >= 0) & (heights_positive < 5)).sum():,} ({((heights_positive >= 0) & (heights_positive < 5)).sum()/len(heights_positive)*100:.1f}%)
Medium (5-15m):            {((heights_positive >= 5) & (heights_positive < 15)).sum():,} ({((heights_positive >= 5) & (heights_positive < 15)).sum()/len(heights_positive)*100:.1f}%)
High (15-30m):             {((heights_positive >= 15) & (heights_positive < 30)).sum():,} ({((heights_positive >= 15) & (heights_positive < 30)).sum()/len(heights_positive)*100:.1f}%)
Very High (>30m):          {(heights_positive >= 30).sum():,} ({(heights_positive >= 30).sum()/len(heights_positive)*100:.1f}%)

{'='*60}
DISTRIBUTION ANALYSIS
{'='*60}

Skewness:                  {self.calculate_skewness(heights_positive):.4f}
Kurtosis:                  {self.calculate_kurtosis(heights_positive):.4f}

Mode (Most Common):        {self.calculate_mode(heights_positive):.2f}m
Range:                     {heights_positive.max() - heights_positive.min():.2f}m

{'='*60}
"""
        self.analytics_text.insert(1.0, stats_text)
    
    def calculate_skewness(self, data):
        """Calculate skewness"""
        n = len(data)
        mean = np.mean(data)
        std = np.std(data, ddof=1)
        return (n / ((n-1) * (n-2))) * np.sum(((data - mean) / std) ** 3)
    
    def calculate_kurtosis(self, data):
        """Calculate kurtosis"""
        n = len(data)
        mean = np.mean(data)
        std = np.std(data, ddof=1)
        return (n * (n+1) / ((n-1) * (n-2) * (n-3))) * np.sum(((data - mean) / std) ** 4) - \
               (3 * (n-1) ** 2 / ((n-2) * (n-3)))
    
    def calculate_mode(self, data):
        """Calculate mode"""
        hist, bin_edges = np.histogram(data, bins=50)
        mode_idx = np.argmax(hist)
        return (bin_edges[mode_idx] + bin_edges[mode_idx + 1]) / 2
    
    def export_results(self):
        """Export all results"""
        if self.current_prediction is None:
            messagebox.showwarning("Warning", "No prediction to export!")
            return
        
        try:
            filepath = filedialog.asksaveasfilename(
                title="Save Results",
                defaultextension=".png",
                filetypes=[("PNG", "*.png"), ("All Files", "*.*")]
            )
            
            if not filepath:
                return
            
            base_path = Path(filepath).with_suffix('')
            
            # Save visualizations
            self.fig_main.savefig(f"{base_path}_main.png", dpi=300, bbox_inches='tight')
            self.fig_analytics.savefig(f"{base_path}_analytics.png", dpi=300, bbox_inches='tight')
            
            # Save height data
            np.save(f"{base_path}_heights.npy", self.current_prediction)
            
            # Save statistics
            heights_positive = self.current_prediction[self.current_prediction > 1]
            stats = {
                'timestamp': datetime.now().isoformat(),
                'image_file': Path(self.image_path).name,
                'model_file': Path(self.checkpoint_path).name,
                'statistics': {
                    'max_height_m': float(self.current_prediction.max()),
                    'mean_height_m': float(heights_positive.mean()) if len(heights_positive) > 0 else 0,
                    'median_height_m': float(np.median(heights_positive)) if len(heights_positive) > 0 else 0,
                    'std_height_m': float(heights_positive.std()) if len(heights_positive) > 0 else 0,
                    'min_height_m': float(heights_positive.min()) if len(heights_positive) > 0 else 0,
                    'building_pixels': int(len(heights_positive)),
                    'total_pixels': int(self.current_prediction.size),
                    'coverage_percent': float(len(heights_positive) / self.current_prediction.size * 100)
                }
            }
            
            with open(f"{base_path}_stats.json", 'w') as f:
                json.dump(stats, f, indent=2)
            
            # Save analytics text
            with open(f"{base_path}_report.txt", 'w') as f:
                f.write(self.analytics_text.get(1.0, tk.END))
            
            messagebox.showinfo("Success", f"Results exported!\n\nSaved files:\n"
                              f"• {base_path.name}_main.png\n"
                              f"• {base_path.name}_analytics.png\n"
                              f"• {base_path.name}_heights.npy\n"
                              f"• {base_path.name}_stats.json\n"
                              f"• {base_path.name}_report.txt")
            
        except Exception as e:
            messagebox.showerror("Error", f"Export failed:\n{str(e)}")
    
    def export_analytics(self):
        """Export only analytics"""
        if self.current_prediction is None:
            messagebox.showwarning("Warning", "No prediction available!")
            return
        
        try:
            filepath = filedialog.asksaveasfilename(
                title="Save Analytics Report",
                defaultextension=".txt",
                filetypes=[("Text File", "*.txt"), ("All Files", "*.*")]
            )
            
            if filepath:
                with open(filepath, 'w') as f:
                    f.write(self.analytics_text.get(1.0, tk.END))
                
                messagebox.showinfo("Success", "Analytics report exported!")
                
        except Exception as e:
            messagebox.showerror("Error", f"Export failed:\n{str(e)}")
    
    def clear_results(self):
        """Clear all results"""
        for ax in self.axes_main:
            ax.clear()
            ax.axis('off')
        self.canvas_main.draw()
        
        for ax in self.axes_analytics.flat:
            ax.clear()
        self.canvas_analytics.draw()
        
        self.analytics_text.delete(1.0, tk.END)
        
        for label in self.stat_labels.values():
            label.config(text="--")
        
        self.current_prediction = None
        self.export_button.config(state=tk.DISABLED)
        self.status_var.set("Results cleared.")
    
    def run(self):
        """Start the GUI"""
        self.root.mainloop()

def main():
    root = tk.Tk()
    app = HeightEstimationGUI(root)
    app.run()

if __name__ == "__main__":
    main()