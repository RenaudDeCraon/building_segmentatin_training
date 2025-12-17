#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Building Height Estimation GUI - Multi-Model Comparison
Load and compare multiple trained model checkpoints
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

class ModelInstance:
    """Container for a loaded model and its metadata"""
    def __init__(self, name, checkpoint_path, model, device, height_stats, config, val_metrics, epoch):
        self.name = name
        self.checkpoint_path = checkpoint_path
        self.model = model
        self.device = device
        self.height_stats = height_stats
        self.config = config
        self.val_metrics = val_metrics
        self.epoch = epoch
        self.prediction = None
        self.color = None  # Will be assigned for visualization

class HeightEstimationGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Building Height Estimation - Multi-Model Comparison")
        self.root.geometry("1800x1000")
        
        # Variables
        self.models = {}  # Dictionary of ModelInstance objects
        self.current_image = None
        self.image_path = None
        
        # Model colors for visualization
        self.model_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8', '#F7DC6F']
        
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
        self.comparison_tab = ttk.Frame(self.notebook)
        self.hyperparams_tab = ttk.Frame(self.notebook)
        
        self.notebook.add(self.main_tab, text="  Prediction  ")
        self.notebook.add(self.analytics_tab, text="  Analytics  ")
        self.notebook.add(self.comparison_tab, text="  Comparison  ")
        self.notebook.add(self.hyperparams_tab, text="  Model Info  ")
        
        # Setup each tab
        self.setup_main_tab()
        self.setup_analytics_tab()
        self.setup_comparison_tab()
        self.setup_hyperparams_tab()
        
        # Status bar (common)
        self.status_var = tk.StringVar(value="Ready. Please load checkpoint(s) and image.")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, 
                              anchor=tk.W, font=('Arial', 9))
        status_bar.pack(side=tk.BOTTOM, fill=tk.X, padx=5, pady=2)
    
    def setup_main_tab(self):
        """Setup main prediction tab"""
        main_frame = ttk.Frame(self.main_tab, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(2, weight=1)
        
        # ===== Model Selection =====
        model_frame = ttk.LabelFrame(main_frame, text="Model Management", padding="10")
        model_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        model_frame.columnconfigure(0, weight=1)
        
        # Model list with scrollbar
        list_frame = ttk.Frame(model_frame)
        list_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 5))
        
        self.model_listbox = tk.Listbox(list_frame, height=3, selectmode=tk.SINGLE)
        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.model_listbox.yview)
        self.model_listbox.config(yscrollcommand=scrollbar.set)
        
        self.model_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Model buttons
        btn_frame = ttk.Frame(model_frame)
        btn_frame.grid(row=1, column=0, pady=5)
        
        ttk.Button(btn_frame, text="➕ Add Model", command=self.add_model).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="🗑️ Remove Selected", command=self.remove_model).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="📊 View Details", command=self.view_model_details).pack(side=tk.LEFT, padx=2)
        
        # ===== Image Selection =====
        file_frame = ttk.LabelFrame(main_frame, text="Image Selection", padding="10")
        file_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        file_frame.columnconfigure(1, weight=1)
        
        ttk.Label(file_frame, text="Input Image:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.image_label = ttk.Label(file_frame, text="No image selected", foreground="gray")
        self.image_label.grid(row=0, column=1, sticky=tk.W, padx=10)
        ttk.Button(file_frame, text="Browse", command=self.load_image).grid(row=0, column=2, padx=5)
        
        # Action buttons
        button_frame = ttk.Frame(file_frame)
        button_frame.grid(row=1, column=0, columnspan=3, pady=10)
        
        self.predict_button = ttk.Button(button_frame, text="🚀 Predict All Models", 
                                        command=self.predict_all, state=tk.DISABLED)
        self.predict_button.grid(row=0, column=0, padx=5)
        
        self.export_button = ttk.Button(button_frame, text="💾 Export Results", 
                                       command=self.export_results, state=tk.DISABLED)
        self.export_button.grid(row=0, column=1, padx=5)
        
        ttk.Button(button_frame, text="🗑️ Clear", command=self.clear_results).grid(row=0, column=2, padx=5)
        
        # ===== Results Visualization =====
        viz_frame = ttk.LabelFrame(main_frame, text="Prediction Results", padding="10")
        viz_frame.grid(row=2, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Create scrollable canvas for multiple model results
        canvas_frame = ttk.Frame(viz_frame)
        canvas_frame.pack(fill=tk.BOTH, expand=True)
        
        self.results_canvas = tk.Canvas(canvas_frame)
        scrollbar_y = ttk.Scrollbar(canvas_frame, orient=tk.VERTICAL, command=self.results_canvas.yview)
        scrollbar_x = ttk.Scrollbar(canvas_frame, orient=tk.HORIZONTAL, command=self.results_canvas.xview)
        
        self.results_canvas.config(yscrollcommand=scrollbar_y.set, xscrollcommand=scrollbar_x.set)
        
        scrollbar_y.pack(side=tk.RIGHT, fill=tk.Y)
        scrollbar_x.pack(side=tk.BOTTOM, fill=tk.X)
        self.results_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.results_frame = ttk.Frame(self.results_canvas)
        self.results_canvas.create_window((0, 0), window=self.results_frame, anchor=tk.NW)
        
        self.results_frame.bind("<Configure>", lambda e: self.results_canvas.configure(
            scrollregion=self.results_canvas.bbox("all")))
    
    def setup_analytics_tab(self):
        """Setup analytics tab - shows all models side by side"""
        analytics_frame = ttk.Frame(self.analytics_tab, padding="10")
        analytics_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create scrollable canvas
        canvas = tk.Canvas(analytics_frame)
        scrollbar = ttk.Scrollbar(analytics_frame, orient=tk.VERTICAL, command=canvas.yview)
        
        canvas.config(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.analytics_container = ttk.Frame(canvas)
        canvas.create_window((0, 0), window=self.analytics_container, anchor=tk.NW)
        
        self.analytics_container.bind("<Configure>", 
                                     lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        
        ttk.Label(self.analytics_container, text="Run predictions to see analytics", 
                 font=('Arial', 14)).pack(pady=50)
    
    def setup_comparison_tab(self):
        """Setup comparison tab - detailed comparison between models"""
        comparison_frame = ttk.Frame(self.comparison_tab, padding="10")
        comparison_frame.pack(fill=tk.BOTH, expand=True)
        
        # Left: Comparison plots
        plot_frame = ttk.LabelFrame(comparison_frame, text="Visual Comparison", padding="10")
        plot_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        self.fig_comparison, self.axes_comparison = plt.subplots(2, 2, figsize=(10, 8))
        self.canvas_comparison = FigureCanvasTkAgg(self.fig_comparison, master=plot_frame)
        self.canvas_comparison.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Right: Comparison statistics
        stats_frame = ttk.LabelFrame(comparison_frame, text="Statistical Comparison", padding="10")
        stats_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))
        
        self.comparison_text = scrolledtext.ScrolledText(stats_frame, width=50, height=40,
                                                        font=('Courier', 10))
        self.comparison_text.pack(fill=tk.BOTH, expand=True)
        
        ttk.Button(stats_frame, text="📊 Export Comparison", 
                  command=self.export_comparison).pack(pady=5)
    
    def setup_hyperparams_tab(self):
        """Setup hyperparameters tab"""
        hyper_frame = ttk.Frame(self.hyperparams_tab, padding="10")
        hyper_frame.pack(fill=tk.BOTH, expand=True)
        
        # Model selector
        selector_frame = ttk.Frame(hyper_frame)
        selector_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(selector_frame, text="Select Model:").pack(side=tk.LEFT, padx=5)
        self.hyperparam_model_var = tk.StringVar()
        self.hyperparam_combo = ttk.Combobox(selector_frame, textvariable=self.hyperparam_model_var,
                                            state='readonly', width=40)
        self.hyperparam_combo.pack(side=tk.LEFT, padx=5)
        self.hyperparam_combo.bind('<<ComboboxSelected>>', self.on_hyperparam_model_select)
        
        # Info display
        self.hyperparam_text = scrolledtext.ScrolledText(hyper_frame, font=('Courier', 10))
        self.hyperparam_text.pack(fill=tk.BOTH, expand=True)
    
    def add_model(self):
        """Add a new model checkpoint"""
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
                device = torch.device("mps")
            elif torch.cuda.is_available():
                device = torch.device("cuda")
            else:
                device = torch.device("cpu")
            
            # Load checkpoint
            checkpoint = torch.load(filepath, map_location=device, weights_only=False)
            
            config = checkpoint.get('config')
            height_stats = checkpoint.get('height_stats')
            val_metrics = checkpoint.get('val_metrics', {})
            epoch = checkpoint.get('epoch', 'Unknown')
            
            if height_stats is None or config is None:
                raise ValueError("Checkpoint missing required information")
            
            # Create model
            model_type = config.model if hasattr(config, 'model') else 'unetplusplus'
            encoder_name = config.encoder if hasattr(config, 'encoder') else 'resnet50'
            
            model_map = {
                'unet': smp.Unet,
                'unetplusplus': smp.UnetPlusPlus,
                'deeplabv3plus': smp.DeepLabV3Plus,
            }
            
            model = model_map[model_type](
                encoder_name=encoder_name,
                encoder_weights=None,
                in_channels=3,
                classes=1,
                activation=None
            )
            
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(device)
            model.eval()
            
            # Create unique name
            model_name = Path(filepath).stem
            counter = 1
            original_name = model_name
            while model_name in self.models:
                model_name = f"{original_name}_{counter}"
                counter += 1
            
            # Assign color
            color_idx = len(self.models) % len(self.model_colors)
            
            # Create ModelInstance
            model_instance = ModelInstance(
                name=model_name,
                checkpoint_path=filepath,
                model=model,
                device=device,
                height_stats=height_stats,
                config=config,
                val_metrics=val_metrics,
                epoch=epoch
            )
            model_instance.color = self.model_colors[color_idx]
            
            # Add to dictionary
            self.models[model_name] = model_instance
            
            # Update UI
            self.model_listbox.insert(tk.END, f"● {model_name} ({model_type}/{encoder_name})")
            self.hyperparam_combo['values'] = list(self.models.keys())
            
            if self.image_path:
                self.predict_button.config(state=tk.NORMAL)
            
            self.status_var.set(f"✅ Added model: {model_name} ({len(self.models)} total)")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model:\n{str(e)}")
            self.status_var.set("Error loading model")
    
    def remove_model(self):
        """Remove selected model"""
        selection = self.model_listbox.curselection()
        if not selection:
            messagebox.showwarning("Warning", "Please select a model to remove")
            return
        
        idx = selection[0]
        model_name = list(self.models.keys())[idx]
        
        # Remove from dictionary
        del self.models[model_name]
        
        # Update UI
        self.model_listbox.delete(idx)
        self.hyperparam_combo['values'] = list(self.models.keys())
        
        if len(self.models) == 0:
            self.predict_button.config(state=tk.DISABLED)
        
        self.status_var.set(f"Removed model: {model_name} ({len(self.models)} remaining)")
    
    def view_model_details(self):
        """Show details of selected model"""
        selection = self.model_listbox.curselection()
        if not selection:
            messagebox.showwarning("Warning", "Please select a model")
            return
        
        idx = selection[0]
        model_name = list(self.models.keys())[idx]
        model_inst = self.models[model_name]
        
        details = f"""
Model: {model_name}
{'='*50}

Architecture: {model_inst.config.model if hasattr(model_inst.config, 'model') else 'Unknown'}
Encoder: {model_inst.config.encoder if hasattr(model_inst.config, 'encoder') else 'Unknown'}
Epoch: {model_inst.epoch}

Validation Metrics:
  MAE: {model_inst.val_metrics.get('mae', 'N/A'):.2f}m
  RMSE: {model_inst.val_metrics.get('rmse', 'N/A'):.2f}m
  R²: {model_inst.val_metrics.get('r2', 'N/A'):.4f}

Height Statistics:
  Range: {model_inst.height_stats['min']:.1f}m - {model_inst.height_stats['max']:.1f}m
  Mean: {model_inst.height_stats['mean']:.2f}m
  Std: {model_inst.height_stats['std']:.2f}m

File: {Path(model_inst.checkpoint_path).name}
        """
        
        messagebox.showinfo(f"Model Details - {model_name}", details)
    
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
            
            self.current_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            self.image_path = filepath
            self.image_label.config(text=Path(filepath).name, foreground="green")
            
            if len(self.models) > 0:
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
    
    def predict_all(self):
        """Run prediction with all loaded models"""
        if not self.models or self.current_image is None:
            messagebox.showwarning("Warning", "Please load models and image!")
            return
        
        try:
            self.status_var.set(f"🔄 Running predictions with {len(self.models)} model(s)...")
            self.root.update()
            
            original_size = self.current_image.shape[:2]
            image_resized = cv2.resize(self.current_image, (256, 256), interpolation=cv2.INTER_LINEAR)
            
            transform = self.get_transforms()
            transformed = transform(image=image_resized)
            image_tensor = transformed['image'].unsqueeze(0)
            
            # Run prediction for each model
            for model_name, model_inst in self.models.items():
                self.status_var.set(f"🔄 Predicting with {model_name}...")
                self.root.update()
                
                image_tensor_device = image_tensor.to(model_inst.device)
                
                with torch.no_grad():
                    pred_normalized = model_inst.model(image_tensor_device)
                
                pred_normalized = pred_normalized.cpu().numpy()[0, 0]
                pred_height = pred_normalized * model_inst.height_stats['std'] + model_inst.height_stats['mean']
                pred_height = np.clip(pred_height, 0, model_inst.height_stats['max'] * 1.2)
                
                pred_height_full = cv2.resize(pred_height, (original_size[1], original_size[0]), 
                                             interpolation=cv2.INTER_LINEAR)
                
                model_inst.prediction = pred_height_full
            
            # Update all visualizations
            self.update_main_results()
            self.update_analytics()
            self.update_comparison()
            
            self.export_button.config(state=tk.NORMAL)
            self.status_var.set("✅ All predictions complete!")
            
        except Exception as e:
            messagebox.showerror("Error", f"Prediction failed:\n{str(e)}")
            self.status_var.set("Error during prediction")
    
    def update_main_results(self):
        """Update main tab with all model predictions"""
        # Clear previous results
        for widget in self.results_frame.winfo_children():
            widget.destroy()
        
        if not self.models:
            return
        
        # Create grid of results
        num_models = len(self.models)
        cols = min(2, num_models)
        
        for idx, (model_name, model_inst) in enumerate(self.models.items()):
            row = idx // cols
            col = idx % cols
            
            # Create frame for this model
            model_frame = ttk.LabelFrame(self.results_frame, text=model_name, padding="10")
            model_frame.grid(row=row, column=col, padx=5, pady=5, sticky=(tk.W, tk.E, tk.N, tk.S))
            
            # Create figure for this model
            fig, axes = plt.subplots(1, 3, figsize=(12, 3))
            
            # Original image
            axes[0].imshow(self.current_image)
            axes[0].set_title('Input', fontsize=10)
            axes[0].axis('off')
            
            # Height map
            im = axes[1].imshow(model_inst.prediction, cmap='plasma', vmin=0, 
                               vmax=model_inst.height_stats['max'])
            axes[1].set_title(f'Max: {model_inst.prediction.max():.1f}m', fontsize=10)
            axes[1].axis('off')
            plt.colorbar(im, ax=axes[1], fraction=0.046)
            
            # Histogram
            heights = model_inst.prediction[model_inst.prediction > 1].flatten()
            if len(heights) > 0:
                axes[2].hist(heights, bins=30, color=model_inst.color, alpha=0.7, edgecolor='black')
                axes[2].axvline(heights.mean(), color='red', linestyle='--', linewidth=2)
                axes[2].set_xlabel('Height (m)', fontsize=9)
                axes[2].set_title(f'Mean: {heights.mean():.1f}m', fontsize=10)
                axes[2].grid(True, alpha=0.3)
            
            fig.tight_layout()
            
            # Embed in tkinter
            canvas = FigureCanvasTkAgg(fig, master=model_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
            # Stats below
            stats_text = f"MAE: {model_inst.val_metrics.get('mae', 0):.2f}m | " \
                        f"RMSE: {model_inst.val_metrics.get('rmse', 0):.2f}m | " \
                        f"Coverage: {(len(heights)/model_inst.prediction.size*100):.1f}%"
            ttk.Label(model_frame, text=stats_text, font=('Arial', 9)).pack()
    
    def update_analytics(self):
        """Update analytics tab with all models"""
        # Clear previous
        for widget in self.analytics_container.winfo_children():
            widget.destroy()
        
        if not self.models or not any(m.prediction is not None for m in self.models.values()):
            ttk.Label(self.analytics_container, text="No predictions available", 
                     font=('Arial', 14)).pack(pady=50)
            return
        
        # Create analytics for each model
        for model_name, model_inst in self.models.items():
            if model_inst.prediction is None:
                continue
            
            frame = ttk.LabelFrame(self.analytics_container, text=f"Analytics - {model_name}", 
                                  padding="10")
            frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
            
            fig, axes = plt.subplots(2, 2, figsize=(14, 7))
            
            heights = model_inst.prediction[model_inst.prediction > 1].flatten()
            
            if len(heights) > 0:
                # Distribution
                axes[0, 0].hist(heights, bins=100, density=True, alpha=0.6, 
                               color=model_inst.color, edgecolor='black')
                axes[0, 0].set_xlabel('Height (m)')
                axes[0, 0].set_ylabel('Density')
                axes[0, 0].set_title('Height Distribution')
                axes[0, 0].grid(True, alpha=0.3)
                
                # Cumulative
                sorted_h = np.sort(heights)
                cumulative = np.arange(1, len(sorted_h) + 1) / len(sorted_h)
                axes[0, 1].plot(sorted_h, cumulative * 100, linewidth=2, color=model_inst.color)
                axes[0, 1].set_xlabel('Height (m)')
                axes[0, 1].set_ylabel('Cumulative %')
                axes[0, 1].set_title('Cumulative Distribution')
                axes[0, 1].grid(True, alpha=0.3)
                
                # Categories
                categories = ['Low\n(0-5m)', 'Med\n(5-15m)', 'High\n(15-30m)', 'V.High\n(>30m)']
                counts = [
                    ((heights >= 0) & (heights < 5)).sum(),
                    ((heights >= 5) & (heights < 15)).sum(),
                    ((heights >= 15) & (heights < 30)).sum(),
                    (heights >= 30).sum()
                ]
                axes[1, 0].bar(categories, counts, color=model_inst.color, alpha=0.7, edgecolor='black')
                axes[1, 0].set_ylabel('Count')
                axes[1, 0].set_title('Height Categories')
                axes[1, 0].grid(True, alpha=0.3, axis='y')
                
                # Box plot
                axes[1, 1].boxplot(heights, vert=True)
                axes[1, 1].set_ylabel('Height (m)')
                axes[1, 1].set_title('Box Plot')
                axes[1, 1].grid(True, alpha=0.3, axis='y')
            
            fig.tight_layout()
            
            canvas = FigureCanvasTkAgg(fig, master=frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def update_comparison(self):
        """Update comparison tab"""
        if len(self.models) < 2 or not any(m.prediction is not None for m in self.models.values()):
            return
        
        # Clear plots
        for ax in self.axes_comparison.flat:
            ax.clear()
        
        # Get models with predictions
        models_with_pred = {name: m for name, m in self.models.items() if m.prediction is not None}
        
        if len(models_with_pred) < 2:
            self.axes_comparison[0, 0].text(0.5, 0.5, 'Need at least 2 predictions', 
                                           ha='center', va='center')
            self.fig_comparison.tight_layout()
            self.canvas_comparison.draw()
            return
        
        # 1. Height map comparison
        for idx, (name, model_inst) in enumerate(models_with_pred.items()):
            if idx < 4:  # Max 4 models in comparison
                row = idx // 2
                col = idx % 2
                im = self.axes_comparison[row, col].imshow(model_inst.prediction, cmap='plasma', 
                                                           vmin=0, vmax=200)
                self.axes_comparison[row, col].set_title(f'{name}\nMax: {model_inst.prediction.max():.1f}m')
                self.axes_comparison[row, col].axis('off')
                plt.colorbar(im, ax=self.axes_comparison[row, col], fraction=0.046)
        
        self.fig_comparison.tight_layout()
        self.canvas_comparison.draw()
        
        # Update comparison statistics
        self.comparison_text.delete(1.0, tk.END)
        
        stats_text = f"""
{'='*60}
MULTI-MODEL COMPARISON REPORT
{'='*60}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Number of Models: {len(models_with_pred)}

"""
        
        # Per-model statistics
        for name, model_inst in models_with_pred.items():
            heights = model_inst.prediction[model_inst.prediction > 1].flatten()
            
            stats_text += f"""
{'='*60}
MODEL: {name}
{'='*60}

Training Metrics:
  Validation MAE:    {model_inst.val_metrics.get('mae', 0):.2f}m
  Validation RMSE:   {model_inst.val_metrics.get('rmse', 0):.2f}m
  R² Score:          {model_inst.val_metrics.get('r2', 0):.4f}

Prediction Statistics:
  Max Height:        {model_inst.prediction.max():.2f}m
  Mean Height:       {heights.mean():.2f}m
  Median Height:     {np.median(heights):.2f}m
  Std Dev:           {heights.std():.2f}m
  Building Coverage: {len(heights)/model_inst.prediction.size*100:.2f}%

"""
        
        # Comparative analysis
        if len(models_with_pred) >= 2:
            stats_text += f"""
{'='*60}
COMPARATIVE ANALYSIS
{'='*60}

"""
            model_list = list(models_with_pred.items())
            for i in range(len(model_list) - 1):
                name1, m1 = model_list[i]
                name2, m2 = model_list[i + 1]
                
                h1 = m1.prediction[m1.prediction > 1]
                h2 = m2.prediction[m2.prediction > 1]
                
                diff_mean = h1.mean() - h2.mean()
                diff_max = m1.prediction.max() - m2.prediction.max()
                
                stats_text += f"""
{name1} vs {name2}:
  Mean Height Difference:    {diff_mean:+.2f}m
  Max Height Difference:     {diff_max:+.2f}m
  MAE Difference:            {m1.val_metrics.get('mae', 0) - m2.val_metrics.get('mae', 0):+.2f}m

"""
        
        self.comparison_text.insert(1.0, stats_text)
    
    def on_hyperparam_model_select(self, event=None):
        """Show hyperparameters for selected model"""
        model_name = self.hyperparam_model_var.get()
        if not model_name or model_name not in self.models:
            return
        
        model_inst = self.models[model_name]
        
        self.hyperparam_text.delete(1.0, tk.END)
        
        info = f"""
{'='*60}
MODEL: {model_name}
{'='*60}

ARCHITECTURE
{'='*60}
Model Type:        {model_inst.config.model if hasattr(model_inst.config, 'model') else 'Unknown'}
Encoder:           {model_inst.config.encoder if hasattr(model_inst.config, 'encoder') else 'Unknown'}
Input Channels:    3 (RGB)
Output Channels:   1 (Height Map)
Training Epoch:    {model_inst.epoch}

PERFORMANCE
{'='*60}
MAE:               {model_inst.val_metrics.get('mae', 'N/A'):.2f} meters
RMSE:              {model_inst.val_metrics.get('rmse', 'N/A'):.2f} meters
R² Score:          {model_inst.val_metrics.get('r2', 'N/A'):.4f}
MAPE:              {model_inst.val_metrics.get('mape', 'N/A'):.2f}%

HYPERPARAMETERS
{'='*60}
Epochs:            {model_inst.config.epochs if hasattr(model_inst.config, 'epochs') else 'N/A'}
Batch Size:        {model_inst.config.batch_size if hasattr(model_inst.config, 'batch_size') else 'N/A'}
Learning Rate:     {model_inst.config.lr if hasattr(model_inst.config, 'lr') else 'N/A'}
Weight Decay:      {model_inst.config.weight_decay if hasattr(model_inst.config, 'weight_decay') else 'N/A'}
Patience:          {model_inst.config.patience if hasattr(model_inst.config, 'patience') else 'N/A'}

DATA NORMALIZATION
{'='*60}
Height Min:        {model_inst.height_stats['min']:.2f}m
Height Max:        {model_inst.height_stats['max']:.2f}m
Height Mean:       {model_inst.height_stats['mean']:.2f}m
Height Std:        {model_inst.height_stats['std']:.2f}m
Total Samples:     {model_inst.height_stats.get('samples', 'N/A'):,}

FILE INFORMATION
{'='*60}
Path:              {model_inst.checkpoint_path}
"""
        
        self.hyperparam_text.insert(1.0, info)
    
    def export_results(self):
        """Export all results"""
        if not any(m.prediction is not None for m in self.models.values()):
            messagebox.showwarning("Warning", "No predictions to export!")
            return
        
        try:
            directory = filedialog.askdirectory(title="Select Export Directory")
            if not directory:
                return
            
            export_dir = Path(directory) / f"export_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            export_dir.mkdir(exist_ok=True)
            
            # Export each model's results
            for name, model_inst in self.models.items():
                if model_inst.prediction is None:
                    continue
                
                model_dir = export_dir / name.replace(' ', '_')
                model_dir.mkdir(exist_ok=True)
                
                # Save prediction
                np.save(model_dir / 'prediction.npy', model_inst.prediction)
                
                # Save statistics
                heights = model_inst.prediction[model_inst.prediction > 1]
                stats = {
                    'model_name': name,
                    'timestamp': datetime.now().isoformat(),
                    'max_height': float(model_inst.prediction.max()),
                    'mean_height': float(heights.mean()) if len(heights) > 0 else 0,
                    'median_height': float(np.median(heights)) if len(heights) > 0 else 0,
                    'coverage': float(len(heights) / model_inst.prediction.size * 100)
                }
                
                with open(model_dir / 'stats.json', 'w') as f:
                    json.dump(stats, f, indent=2)
            
            # Save comparison report
            with open(export_dir / 'comparison_report.txt', 'w') as f:
                f.write(self.comparison_text.get(1.0, tk.END))
            
            messagebox.showinfo("Success", f"Results exported to:\n{export_dir}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Export failed:\n{str(e)}")
    
    def export_comparison(self):
        """Export comparison report"""
        try:
            filepath = filedialog.asksaveasfilename(
                title="Save Comparison Report",
                defaultextension=".txt",
                filetypes=[("Text File", "*.txt"), ("All Files", "*.*")]
            )
            
            if filepath:
                with open(filepath, 'w') as f:
                    f.write(self.comparison_text.get(1.0, tk.END))
                
                messagebox.showinfo("Success", "Comparison report exported!")
                
        except Exception as e:
            messagebox.showerror("Error", f"Export failed:\n{str(e)}")
    
    def clear_results(self):
        """Clear all predictions"""
        for model_inst in self.models.values():
            model_inst.prediction = None
        
        for widget in self.results_frame.winfo_children():
            widget.destroy()
        
        for widget in self.analytics_container.winfo_children():
            widget.destroy()
        
        ttk.Label(self.analytics_container, text="Run predictions to see analytics", 
                 font=('Arial', 14)).pack(pady=50)
        
        for ax in self.axes_comparison.flat:
            ax.clear()
        self.canvas_comparison.draw()
        
        self.comparison_text.delete(1.0, tk.END)
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