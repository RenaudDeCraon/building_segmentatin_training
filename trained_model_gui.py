"""
GUI for Building Segmentation with PNG support using your trained model
Enhanced version that works with PNG images and your custom trained model
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
import threading
import numpy as np
import cv2
import torch
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
import warnings
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Suppress warnings
warnings.filterwarnings("ignore")

class PNGBuildingSegmentationGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("PNG Building Segmentation Tool - Trained Model")
        self.root.geometry("1200x800")
        self.root.resizable(True, True)
        
        # Variables
        self.input_folder_path = tk.StringVar()
        self.output_directory = tk.StringVar()
        self.model_path = tk.StringVar()
        self.confidence_thresh = tk.DoubleVar(value=0.5)
        self.min_area = tk.IntVar(value=100)
        self.model_type = tk.StringVar(value="unet")
        self.encoder_type = tk.StringVar(value="resnet34")
        self.post_processing = tk.BooleanVar(value=True)
        
        # Model variables
        self.model = None
        self.device = None
        self.transform = None
        
        # File management
        self.image_files = []
        self.selected_files = {}
        self.current_image = None
        self.current_mask = None
        
        # Supported formats
        self.supported_formats = ('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp')
        
        self.setup_scrollable_ui()
        
    def setup_scrollable_ui(self):
        # Create main canvas and scrollbar
        self.canvas = tk.Canvas(self.root, highlightthickness=0)
        self.scrollbar = ttk.Scrollbar(self.root, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = ttk.Frame(self.canvas)
        
        # Configure scrollable frame
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )
        
        # Create window in canvas
        self.canvas_window = self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        
        # Configure canvas scrolling
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        
        # Pack canvas and scrollbar
        self.canvas.pack(side="left", fill="both", expand=True, padx=(10, 0), pady=10)
        self.scrollbar.pack(side="right", fill="y", padx=(0, 10), pady=10)
        
        # Bind mousewheel to canvas
        self.canvas.bind("<MouseWheel>", self._on_mousewheel)
        self.root.bind("<MouseWheel>", self._on_mousewheel)
        
        # Bind canvas resize
        self.canvas.bind("<Configure>", self._on_canvas_configure)
        
        # Set up the UI content
        self.setup_ui_content()
        
        # Configure focus and key bindings for scrolling
        self.canvas.focus_set()
        
    def _on_mousewheel(self, event):
        self.canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        
    def _on_canvas_configure(self, event):
        canvas_width = event.width
        self.canvas.itemconfig(self.canvas_window, width=canvas_width)
        
    def setup_ui_content(self):
        # Title
        title_label = ttk.Label(self.scrollable_frame, text="PNG Building Segmentation Tool", 
                               font=("Arial", 16, "bold"))
        title_label.pack(pady=(0, 20))
        
        # Model Configuration Section
        model_frame = ttk.LabelFrame(self.scrollable_frame, text="Model Configuration", padding=10)
        model_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Model path
        ttk.Label(model_frame, text="Trained Model Checkpoint:").pack(anchor=tk.W)
        model_path_frame = ttk.Frame(model_frame)
        model_path_frame.pack(fill=tk.X, pady=(5, 10))
        
        ttk.Entry(model_path_frame, textvariable=self.model_path, width=60).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(model_path_frame, text="Browse", command=self.browse_model).pack(side=tk.RIGHT, padx=(5, 0))
        
        # Model architecture
        arch_frame = ttk.Frame(model_frame)
        arch_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(arch_frame, text="Model Type:").pack(side=tk.LEFT)
        model_combo = ttk.Combobox(arch_frame, textvariable=self.model_type, 
                                  values=["unet", "unetplusplus", "deeplabv3plus"], 
                                  state="readonly", width=15)
        model_combo.pack(side=tk.LEFT, padx=(10, 20))
        
        ttk.Label(arch_frame, text="Encoder:").pack(side=tk.LEFT)
        encoder_combo = ttk.Combobox(arch_frame, textvariable=self.encoder_type, 
                                   values=["resnet34", "resnet50", "efficientnet-b0"], 
                                   state="readonly", width=15)
        encoder_combo.pack(side=tk.LEFT, padx=(10, 0))
        
        # Load model button
        ttk.Button(model_frame, text="Load Model", command=self.load_model).pack(pady=(10, 0))
        
        # Input Folder Section
        input_frame = ttk.LabelFrame(self.scrollable_frame, text="Input Images", padding=10)
        input_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(input_frame, text="Select Folder with Images (PNG, JPG, TIFF):").pack(anchor=tk.W)
        input_path_frame = ttk.Frame(input_frame)
        input_path_frame.pack(fill=tk.X, pady=(5, 0))
        
        ttk.Entry(input_path_frame, textvariable=self.input_folder_path, width=60).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(input_path_frame, text="Browse", command=self.browse_input_folder).pack(side=tk.RIGHT, padx=(5, 0))
        
        # File Selection Section
        self.file_frame = ttk.LabelFrame(self.scrollable_frame, text="Image Files", padding=10)
        self.file_frame.pack(fill=tk.X, pady=(0, 10))
        
        # File count and selection controls
        file_controls_frame = ttk.Frame(self.file_frame)
        file_controls_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.file_count_label = ttk.Label(file_controls_frame, text="No images found")
        self.file_count_label.pack(side=tk.LEFT)
        
        ttk.Button(file_controls_frame, text="Select All", command=self.select_all_files).pack(side=tk.RIGHT, padx=(5, 0))
        ttk.Button(file_controls_frame, text="Deselect All", command=self.deselect_all_files).pack(side=tk.RIGHT, padx=(5, 0))
        
        # Create frame for file list with scrollbar
        self.file_list_frame = ttk.Frame(self.file_frame)
        self.file_list_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create canvas for file list
        self.file_canvas = tk.Canvas(self.file_list_frame, height=150, highlightthickness=1, highlightbackground="gray")
        self.file_scrollbar = ttk.Scrollbar(self.file_list_frame, orient="vertical", command=self.file_canvas.yview)
        self.file_content_frame = ttk.Frame(self.file_canvas)
        
        self.file_content_frame.bind(
            "<Configure>",
            lambda e: self.file_canvas.configure(scrollregion=self.file_canvas.bbox("all"))
        )
        
        self.file_canvas.create_window((0, 0), window=self.file_content_frame, anchor="nw")
        self.file_canvas.configure(yscrollcommand=self.file_scrollbar.set)
        
        self.file_canvas.pack(side="left", fill="both", expand=True)
        self.file_scrollbar.pack(side="right", fill="y")
        
        # Bind mousewheel to file canvas
        self.file_canvas.bind("<MouseWheel>", self._on_file_mousewheel)
        
        # Output Section
        output_frame = ttk.LabelFrame(self.scrollable_frame, text="Output Configuration", padding=10)
        output_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(output_frame, text="Output Directory:").pack(anchor=tk.W)
        output_dir_frame = ttk.Frame(output_frame)
        output_dir_frame.pack(fill=tk.X, pady=(5, 0))
        
        ttk.Entry(output_dir_frame, textvariable=self.output_directory, width=60).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(output_dir_frame, text="Browse", command=self.browse_output_directory).pack(side=tk.RIGHT, padx=(5, 0))
        
        # Output format options
        format_frame = ttk.Frame(output_frame)
        format_frame.pack(fill=tk.X, pady=(10, 0))
        
        ttk.Label(format_frame, text="Save:").pack(side=tk.LEFT)
        self.save_overlay = tk.BooleanVar(value=True)
        self.save_mask = tk.BooleanVar(value=True)
        self.save_stats = tk.BooleanVar(value=True)
        
        ttk.Checkbutton(format_frame, text="Overlay Image", variable=self.save_overlay).pack(side=tk.LEFT, padx=(10, 10))
        ttk.Checkbutton(format_frame, text="Binary Mask", variable=self.save_mask).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Checkbutton(format_frame, text="Statistics", variable=self.save_stats).pack(side=tk.LEFT)
        
        # Parameters Section
        params_frame = ttk.LabelFrame(self.scrollable_frame, text="Segmentation Parameters", padding=10)
        params_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Confidence threshold
        conf_frame = ttk.Frame(params_frame)
        conf_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(conf_frame, text="Confidence Threshold:").pack(side=tk.LEFT)
        conf_scale = ttk.Scale(conf_frame, from_=0.1, to=0.9, variable=self.confidence_thresh, 
                              orient=tk.HORIZONTAL, length=200, command=self.on_parameter_change)
        conf_scale.pack(side=tk.LEFT, padx=(10, 10))
        self.conf_label = ttk.Label(conf_frame, text="0.50")
        self.conf_label.pack(side=tk.LEFT)
        
        # Min area and post-processing
        options_frame = ttk.Frame(params_frame)
        options_frame.pack(fill=tk.X)
        
        ttk.Label(options_frame, text="Min Building Area (pixels):").pack(side=tk.LEFT)
        ttk.Entry(options_frame, textvariable=self.min_area, width=10).pack(side=tk.LEFT, padx=(10, 20))
        
        ttk.Checkbutton(options_frame, text="Apply Post-Processing", variable=self.post_processing).pack(side=tk.LEFT)
        
        # Preview Section
        preview_frame = ttk.LabelFrame(self.scrollable_frame, text="Preview", padding=10)
        preview_frame.pack(fill=tk.X, pady=(0, 10))
        
        preview_controls = ttk.Frame(preview_frame)
        preview_controls.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(preview_controls, text="🔍 Preview Selected", command=self.preview_image).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(preview_controls, text="📊 Show Statistics", command=self.show_preview_stats).pack(side=tk.LEFT)
        
        # Progress Section
        progress_frame = ttk.LabelFrame(self.scrollable_frame, text="Progress", padding=10)
        progress_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.progress_var = tk.StringVar(value="Ready - Load model first")
        self.progress_label = ttk.Label(progress_frame, textvariable=self.progress_var)
        self.progress_label.pack(anchor=tk.W)
        
        self.progress_bar = ttk.Progressbar(progress_frame, mode='determinate')
        self.progress_bar.pack(fill=tk.X, pady=(5, 0))
        
        # Buttons
        button_frame = ttk.Frame(self.scrollable_frame)
        button_frame.pack(fill=tk.X, pady=(10, 0))
        
        self.segment_button = ttk.Button(button_frame, text="🚀 Start Batch Segmentation", 
                                        command=self.start_segmentation, state="disabled")
        self.segment_button.pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Button(button_frame, text="🔄 Reset", command=self.reset_form).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(button_frame, text="❌ Exit", command=self.root.quit).pack(side=tk.RIGHT)
        
        # Update threshold label
        self.confidence_thresh.trace('w', self.update_conf_label)
        
    def _on_file_mousewheel(self, event):
        self.file_canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        
    def on_parameter_change(self, *args):
        """Called when parameters change"""
        pass
        
    def update_conf_label(self, *args):
        self.conf_label.config(text=f"{self.confidence_thresh.get():.2f}")
        
    def browse_model(self):
        filename = filedialog.askopenfilename(
            title="Select Trained Model Checkpoint",
            filetypes=[
                ("PyTorch files", "*.pth"),
                ("All files", "*.*")
            ]
        )
        if filename:
            self.model_path.set(filename)
            
    def browse_input_folder(self):
        folder = filedialog.askdirectory(title="Select Folder with Images")
        if folder:
            self.input_folder_path.set(folder)
            self.scan_folder_for_images()
            
    def browse_output_directory(self):
        directory = filedialog.askdirectory(title="Select Output Directory")
        if directory:
            self.output_directory.set(directory)
            
    def scan_folder_for_images(self):
        """Scan the selected folder for supported image formats"""
        folder_path = self.input_folder_path.get()
        if not folder_path or not os.path.exists(folder_path):
            return
            
        # Find all supported image files
        self.image_files = []
        for file in os.listdir(folder_path):
            if file.lower().endswith(self.supported_formats):
                self.image_files.append(file)
                
        self.image_files.sort()  # Sort alphabetically
        
        # Initialize selection dictionary (all selected by default)
        self.selected_files = {file: tk.BooleanVar(value=True) for file in self.image_files}
        
        # Update UI
        self.update_file_list()
        self.update_file_count()
        
    def update_file_list(self):
        """Update the file list display"""
        # Clear existing widgets
        for widget in self.file_content_frame.winfo_children():
            widget.destroy()
            
        if not self.image_files:
            ttk.Label(self.file_content_frame, text="No supported images found in selected folder", 
                     font=("Arial", 10, "italic")).pack(pady=20)
            return
            
        # Create checkboxes for each file
        for i, file in enumerate(self.image_files):
            file_frame = ttk.Frame(self.file_content_frame)
            file_frame.pack(fill=tk.X, padx=5, pady=2)
            
            checkbox = ttk.Checkbutton(file_frame, text=file, variable=self.selected_files[file],
                                     command=self.update_file_count)
            checkbox.pack(side=tk.LEFT, anchor=tk.W)
            
        # Update canvas scroll region
        self.file_content_frame.update_idletasks()
        self.file_canvas.configure(scrollregion=self.file_canvas.bbox("all"))
        
    def update_file_count(self):
        """Update the file count display"""
        if not self.image_files:
            self.file_count_label.config(text="No images found")
            return
            
        total_files = len(self.image_files)
        selected_files = sum(1 for var in self.selected_files.values() if var.get())
        
        self.file_count_label.config(text=f"Found {total_files} images, {selected_files} selected")
        
    def select_all_files(self):
        """Select all files"""
        for var in self.selected_files.values():
            var.set(True)
        self.update_file_count()
        
    def deselect_all_files(self):
        """Deselect all files"""
        for var in self.selected_files.values():
            var.set(False)
        self.update_file_count()
        
    def load_model(self):
        """Load the trained model"""
        if not self.model_path.get() or not os.path.exists(self.model_path.get()):
            messagebox.showerror("Error", "Please select a valid model checkpoint file.")
            return
            
        try:
            self.progress_var.set("Loading model...")
            
            # Setup device
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
            
            # Create model architecture
            model_map = {
                'unet': smp.Unet,
                'unetplusplus': smp.UnetPlusPlus,
                'deeplabv3plus': smp.DeepLabV3Plus,
            }
            
            self.model = model_map[self.model_type.get()](
                encoder_name=self.encoder_type.get(),
                encoder_weights=None,  # We're loading pretrained weights
                in_channels=3,
                classes=1,
                activation=None
            )
            
            # Load checkpoint with PyTorch 2.6 compatibility
            try:
                checkpoint = torch.load(self.model_path.get(), map_location=self.device, weights_only=True)
            except:
                checkpoint = torch.load(self.model_path.get(), map_location=self.device, weights_only=False)
            
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)
            self.model.eval()
            
            # Setup transforms
            self.transform = A.Compose([
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ])
            
            self.progress_var.set(f"Model loaded successfully on {self.device}")
            self.segment_button.config(state="normal")
            
            # Show model info
            best_iou = checkpoint.get('best_iou', 'Unknown')
            messagebox.showinfo("Model Loaded", 
                              f"Model loaded successfully!\n\n"
                              f"Architecture: {self.model_type.get()}\n"
                              f"Encoder: {self.encoder_type.get()}\n"
                              f"Device: {self.device}\n"
                              f"Best IoU: {best_iou}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model:\n{str(e)}")
            self.progress_var.set("Failed to load model")
            
    def load_image(self, image_path):
        """Load image in various formats"""
        try:
            # Try loading with different methods based on extension
            ext = os.path.splitext(image_path)[1].lower()
            
            if ext in ['.tif', '.tiff']:
                # Use rasterio for TIFF files
                import rasterio
                from rasterio.plot import reshape_as_image
                
                with rasterio.open(image_path) as src:
                    img_array = src.read()
                    image = reshape_as_image(img_array)
                    
                    # Handle different band combinations
                    if image.shape[2] > 3:
                        image = image[:, :, :3]
                    elif image.shape[2] == 1:
                        image = np.repeat(image, 3, axis=2)
            else:
                # Use OpenCV for other formats
                image = cv2.imread(image_path)
                if image is None:
                    raise ValueError(f"Could not load image: {image_path}")
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Normalize to 0-255 if needed
            if image.dtype != np.uint8:
                image = ((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)
            
            return image
            
        except Exception as e:
            raise ValueError(f"Error loading image {image_path}: {str(e)}")
    
    def predict_image(self, image):
        """Run model prediction on image"""
        # Apply transforms
        transformed = self.transform(image=image)
        img_tensor = transformed['image'].unsqueeze(0).to(self.device)
        
        # Predict
        with torch.no_grad():
            output = self.model(img_tensor)
            prediction = torch.sigmoid(output).cpu().numpy()[0, 0]
        
        # Apply threshold
        binary_mask = (prediction > self.confidence_thresh.get()).astype(np.uint8)
        
        # Apply post-processing if enabled
        if self.post_processing.get():
            binary_mask = self.apply_post_processing(image, binary_mask)
        
        # Remove small areas
        if self.min_area.get() > 0:
            # Find connected components
            num_labels, labels = cv2.connectedComponents(binary_mask)
            for i in range(1, num_labels):
                component_mask = (labels == i)
                if np.sum(component_mask) < self.min_area.get():
                    binary_mask[component_mask] = 0
        
        return binary_mask, prediction
    
    def apply_post_processing(self, image, mask):
        """Apply post-processing to remove false positives"""
        # Simple post-processing to remove linear features and road-like objects
        num_labels, labels = cv2.connectedComponents(mask)
        filtered_mask = np.zeros_like(mask)
        
        for i in range(1, num_labels):
            component = (labels == i).astype(np.uint8)
            
            # Calculate aspect ratio
            contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue
                
            contour = max(contours, key=cv2.contourArea)
            rect = cv2.minAreaRect(contour)
            width, height = rect[1]
            
            if width > 0 and height > 0:
                aspect_ratio = max(width, height) / min(width, height)
                
                # Keep only if aspect ratio is reasonable for buildings (not too linear)
                if aspect_ratio <= 8.0:  # Remove very linear features
                    filtered_mask[component > 0] = 1
            else:
                filtered_mask[component > 0] = 1
        
        return filtered_mask
    
    def preview_image(self):
        """Preview segmentation on selected image"""
        if not self.model:
            messagebox.showerror("Error", "Please load a model first.")
            return
            
        selected_files = [file for file, var in self.selected_files.items() if var.get()]
        if not selected_files:
            messagebox.showerror("Error", "Please select at least one image.")
            return
            
        # Use first selected file for preview
        image_path = os.path.join(self.input_folder_path.get(), selected_files[0])
        
        try:
            # Load and predict
            image = self.load_image(image_path)
            mask, confidence = self.predict_image(image)
            
            # Store for statistics
            self.current_image = image
            self.current_mask = mask
            
            # Create preview window
            self.show_preview_window(image, mask, confidence, selected_files[0])
            
        except Exception as e:
            messagebox.showerror("Error", f"Preview failed: {str(e)}")
    
    def show_preview_window(self, image, mask, confidence, filename):
        """Show preview window with results"""
        preview_window = tk.Toplevel(self.root)
        preview_window.title(f"Preview: {filename}")
        preview_window.geometry("1000x600")
        
        # Create matplotlib figure
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image
        axes[0].imshow(image)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # Confidence map
        im1 = axes[1].imshow(confidence, cmap='viridis', vmin=0, vmax=1)
        axes[1].set_title(f'Confidence Map\n(Threshold: {self.confidence_thresh.get():.2f})')
        axes[1].axis('off')
        plt.colorbar(im1, ax=axes[1])
        
        # Result overlay
        overlay = image.copy()
        overlay[mask == 1] = [0, 255, 0]  # Green for buildings
        axes[2].imshow(overlay)
        
        # Count buildings
        num_labels, _ = cv2.connectedComponents(mask)
        building_count = num_labels - 1
        
        axes[2].set_title(f'Detected Buildings\n({building_count} buildings found)')
        axes[2].axis('off')
        
        plt.tight_layout()
        
        # Embed in tkinter
        canvas = FigureCanvasTkAgg(fig, preview_window)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
    def show_preview_stats(self):
        """Show statistics for current preview"""
        if self.current_mask is None:
            messagebox.showinfo("Info", "Please run a preview first.")
            return
            
        # Calculate statistics
        num_labels, labels = cv2.connectedComponents(self.current_mask)
        building_count = num_labels - 1
        total_building_area = np.sum(self.current_mask)
        image_area = self.current_mask.size
        coverage_percent = (total_building_area / image_area) * 100
        
        # Building size statistics
        if building_count > 0:
            building_sizes = []
            for i in range(1, num_labels):
                size = np.sum(labels == i)
                building_sizes.append(size)
            
            avg_size = np.mean(building_sizes)
            max_size = np.max(building_sizes)
            min_size = np.min(building_sizes)
        else:
            avg_size = max_size = min_size = 0
        
        stats_text = f"""
Building Detection Statistics:

📊 Overall:
• Buildings found: {building_count}
• Total building area: {total_building_area:,} pixels
• Building coverage: {coverage_percent:.1f}%

📏 Building Sizes:
• Average size: {avg_size:.0f} pixels
• Largest building: {max_size:,} pixels
• Smallest building: {min_size:,} pixels

⚙️ Settings Used:
• Confidence threshold: {self.confidence_thresh.get():.2f}
• Minimum area: {self.min_area.get()} pixels
• Post-processing: {"Enabled" if self.post_processing.get() else "Disabled"}
        """
        
        messagebox.showinfo("Detection Statistics", stats_text)
    
    def start_segmentation(self):
        """Start the batch segmentation process"""
        if not self.validate_inputs():
            return
            
        self.segment_button.config(state="disabled")
        
        # Get selected files
        selected_files = [file for file, var in self.selected_files.items() if var.get()]
        
        # Setup progress bar
        self.progress_bar.config(mode='determinate', maximum=len(selected_files))
        self.progress_bar['value'] = 0
        
        # Run segmentation in separate thread
        thread = threading.Thread(target=self.run_batch_segmentation, args=(selected_files,))
        thread.daemon = True
        thread.start()
        
    def validate_inputs(self):
        """Validate all inputs"""
        if self.model is None:
            messagebox.showerror("Error", "Please load a model first.")
            return False
            
        if not self.input_folder_path.get():
            messagebox.showerror("Error", "Please select an input folder.")
            return False
            
        if not self.output_directory.get():
            messagebox.showerror("Error", "Please select an output directory.")
            return False
            
        if not os.path.exists(self.output_directory.get()):
            messagebox.showerror("Error", "Output directory does not exist.")
            return False
            
        selected_count = sum(1 for var in self.selected_files.values() if var.get())
        if selected_count == 0:
            messagebox.showerror("Error", "Please select at least one image to process.")
            return False
            
        return True
        
    def run_batch_segmentation(self, selected_files):
        """Run segmentation on all selected images"""
        try:
            total_files = len(selected_files)
            successful_files = 0
            total_buildings = 0
            failed_files = []
            
            for i, filename in enumerate(selected_files):
                try:
                    self.update_progress(f"Processing {filename} ({i+1}/{total_files})...")
                    
                    # Process image
                    input_path = os.path.join(self.input_folder_path.get(), filename)
                    image = self.load_image(input_path)
                    
                    # Predict
                    mask, confidence = self.predict_image(image)
                    
                    # Count buildings (connected components)
                    num_labels, _ = cv2.connectedComponents(mask)
                    building_count = num_labels - 1
                    total_buildings += building_count
                    
                    # Save results
                    base_name = os.path.splitext(filename)[0]
                    self.save_results(image, mask, confidence, base_name, building_count)
                    
                    successful_files += 1
                    
                except Exception as e:
                    failed_files.append(f"{filename}: {str(e)}")
                    print(f"Error processing {filename}: {e}")
                
                # Update progress
                self.root.after(0, lambda: self.progress_bar.config(value=i+1))
            
            # Show results
            result_message = f"Batch segmentation completed!\n\n"
            result_message += f"Successfully processed: {successful_files}/{total_files} files\n"
            result_message += f"Total buildings detected: {total_buildings}\n"
            result_message += f"Results saved to: {self.output_directory.get()}\n"
            
            if failed_files:
                result_message += f"\nFailed files:\n" + "\n".join(failed_files[:5])
                if len(failed_files) > 5:
                    result_message += f"\n... and {len(failed_files) - 5} more"
            
            self.root.after(0, lambda: messagebox.showinfo("Processing Complete", result_message))
            self.update_progress("Batch segmentation completed!")
            
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Error", f"Batch segmentation failed:\n{str(e)}"))
            self.update_progress("Error during batch segmentation")
            
        finally:
            self.root.after(0, lambda: self.segment_button.config(state="normal"))
            
    def save_results(self, image, mask, confidence, base_name, building_count):
        """Save segmentation results in various formats"""
        output_dir = self.output_directory.get()
        
        # Save overlay image (buildings in green)
        if self.save_overlay.get():
            overlay = image.copy()
            overlay[mask == 1] = [0, 255, 0]  # Green for buildings
            overlay_path = os.path.join(output_dir, f"{base_name}_overlay.png")
            cv2.imwrite(overlay_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        
        # Save binary mask
        if self.save_mask.get():
            mask_path = os.path.join(output_dir, f"{base_name}_mask.png")
            mask_255 = (mask * 255).astype(np.uint8)
            cv2.imwrite(mask_path, mask_255)
        
        # Save statistics
        if self.save_stats.get():
            stats = self.calculate_image_statistics(mask, building_count)
            stats_path = os.path.join(output_dir, f"{base_name}_stats.txt")
            
            with open(stats_path, 'w') as f:
                f.write(f"Building Detection Statistics for {base_name}\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"Buildings detected: {building_count}\n")
                f.write(f"Total building area: {stats['total_area']:,} pixels\n")
                f.write(f"Building coverage: {stats['coverage']:.2f}%\n")
                f.write(f"Average building size: {stats['avg_size']:.0f} pixels\n")
                f.write(f"Largest building: {stats['max_size']:,} pixels\n")
                f.write(f"Smallest building: {stats['min_size']:,} pixels\n\n")
                f.write(f"Processing parameters:\n")
                f.write(f"- Confidence threshold: {self.confidence_thresh.get():.2f}\n")
                f.write(f"- Minimum area: {self.min_area.get()} pixels\n")
                f.write(f"- Post-processing: {'Enabled' if self.post_processing.get() else 'Disabled'}\n")
    
    def calculate_image_statistics(self, mask, building_count):
        """Calculate statistics for an image"""
        total_area = np.sum(mask)
        image_area = mask.size
        coverage = (total_area / image_area) * 100
        
        if building_count > 0:
            num_labels, labels = cv2.connectedComponents(mask)
            building_sizes = []
            for i in range(1, num_labels):
                size = np.sum(labels == i)
                building_sizes.append(size)
            
            avg_size = np.mean(building_sizes)
            max_size = np.max(building_sizes)
            min_size = np.min(building_sizes)
        else:
            avg_size = max_size = min_size = 0
        
        return {
            'total_area': total_area,
            'coverage': coverage,
            'avg_size': avg_size,
            'max_size': max_size,
            'min_size': min_size
        }
    
    def update_progress(self, message):
        self.root.after(0, lambda: self.progress_var.set(message))
        
    def reset_form(self):
        """Reset all form fields"""
        self.input_folder_path.set("")
        self.output_directory.set("")
        self.model_path.set("")
        self.confidence_thresh.set(0.5)
        self.min_area.set(100)
        self.post_processing.set(True)
        self.save_overlay.set(True)
        self.save_mask.set(True)
        self.save_stats.set(True)
        
        self.model = None
        self.device = None
        self.transform = None
        self.current_image = None
        self.current_mask = None
        
        self.image_files = []
        self.selected_files = {}
        self.update_file_list()
        self.update_file_count()
        
        self.progress_var.set("Ready - Load model first")
        self.progress_bar.config(value=0)
        self.segment_button.config(state="disabled")

def main():
    root = tk.Tk()
    app = PNGBuildingSegmentationGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()