# 🔥 Thermal Human Detection using YOLOv8

A comprehensive deep learning system for detecting humans in thermal imagery using YOLOv8. This project includes model training, evaluation, real-time video processing, and deployment capabilities.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/pytorch-2.0+-orange.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

---

## 📋 Table of Contents

- [Features](#features)
- [System Requirements](#system-requirements)
- [Installation](#installation)
- [Dataset Preparation](#dataset-preparation)
- [Usage Guide](#usage-guide)
- [Project Structure](#project-structure)
- [Results](#results)
- [Troubleshooting](#troubleshooting)
- [Credits](#credits)

---

## ✨ Features

- 🎯 **Human Detection** in thermal images with high accuracy
- 📊 **Comprehensive Training Pipeline** with hyperparameter tuning
- 📈 **Performance Visualization** (loss curves, mAP, precision/recall)
- 🎥 **Real-time Video Processing** with frame-by-frame detection
- 📸 **Batch Image Processing** for multiple images
- ⚡ **Model Benchmarking** (FPS, inference time)
- 🔄 **Model Comparison** between YOLOv8 variants
- 📦 **Model Export** (ONNX, TorchScript, TFLite)
- 🖥️ **Jupyter Notebook Compatible** for interactive development

---

## 💻 System Requirements

### Minimum Requirements
- **OS**: Windows 10/11, macOS 10.15+, or Linux (Ubuntu 18.04+)
- **RAM**: 8 GB
- **Storage**: 10 GB free space
- **Python**: 3.8 or higher

### Recommended Requirements
- **GPU**: NVIDIA GPU with 6+ GB VRAM (CUDA support)
- **RAM**: 16 GB or more
- **Storage**: 20 GB free space (for datasets and outputs)
- **CUDA**: 11.7 or higher (for GPU acceleration)

### Check Your System
```bash
# Check Python version
python --version

# Check if CUDA is available (for GPU)
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"
```

---

## 🚀 Installation

### Step 1: Install Jupyter Lab

#### Option A: Using Anaconda (Recommended)
```bash
# Download and install Anaconda from: https://www.anaconda.com/download

# Create a new environment
conda create -n thermal-detection python=3.9
conda activate thermal-detection

# Install Jupyter Lab
conda install -c conda-forge jupyterlab
```

#### Option B: Using pip
```bash
# Install Jupyter Lab
pip install jupyterlab

# Launch Jupyter Lab
jupyter lab
```

### Step 2: Install Required Packages

Open a terminal in Jupyter Lab or run in the first notebook cell:

```bash
# Install core packages
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install YOLO and dependencies
pip install ultralytics supervision huggingface_hub

# Install visualization and utilities
pip install opencv-python pandas matplotlib seaborn numpy

# Install additional tools
pip install ipywidgets notebook ipykernel
```

### Step 3: Verify Installation

Create a new cell and run:

```python
import torch
import cv2
from ultralytics import YOLO
import supervision as sv

print("✓ All packages installed successfully!")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda if torch.cuda.is_available() else 'N/A'}")
```

---

## 📁 Dataset Preparation

### Dataset Structure

Your dataset should follow this structure:

```
thermal_dataset/
├── data.yaml           # Dataset configuration file
├── train/
│   ├── images/
│   │   ├── image001.jpg
│   │   ├── image002.jpg
│   │   └── ...
│   └── labels/
│       ├── image001.txt
│       ├── image002.txt
│       └── ...
├── valid/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

### Creating `data.yaml`

Create a file named `data.yaml` in your project root:

```yaml
# Dataset paths
path: ./thermal_dataset  # Root directory
train: train/images      # Training images
val: valid/images        # Validation images
test: test/images        # Test images (optional)

# Classes
nc: 1                    # Number of classes
names: ['person']        # Class names
```

### Label Format (YOLO format)

Each `.txt` file should contain bounding boxes in YOLO format:

```
class_id center_x center_y width height
```

Example (`image001.txt`):
```
0 0.5 0.5 0.3 0.4
0 0.7 0.3 0.2 0.3
```

Where:
- `class_id`: 0 (for person)
- `center_x, center_y`: Normalized center coordinates (0-1)
- `width, height`: Normalized box dimensions (0-1)

### Quick Dataset Check

Run this in a notebook cell:

```python
from pathlib import Path

data_path = Path("thermal_dataset")
train_images = list((data_path / "train/images").glob("*.jpg"))
train_labels = list((data_path / "train/labels").glob("*.txt"))

print(f"Training images: {len(train_images)}")
print(f"Training labels: {len(train_labels)}")
print(f"Match: {len(train_images) == len(train_labels)}")
```

---

## 📖 Usage Guide

### 1. Launch Jupyter Lab

```bash
# Activate your environment (if using conda)
conda activate thermal-detection

# Navigate to project directory
cd /path/to/your/project

# Launch Jupyter Lab
jupyter lab
```

### 2. Create a New Notebook

1. Click **File** → **New** → **Notebook**
2. Copy the code from the thermal detection script into cells
3. Run cells sequentially

### 3. Basic Workflow

#### A. Load Model

```python
from ultralytics import YOLO

# Load pre-trained YOLOv8 model
model = YOLO('yolov8n.pt')
print("✓ Model loaded successfully!")
```

#### B. Single Image Inference

```python
# Test on a single image
detections, annotated = inference_single_image(
    image_path="path/to/your/thermal_image.jpg",
    model=model,
    conf=0.6
)
```

#### C. Train Custom Model

```python
# Train on your thermal dataset
trained_model, results = train_model(
    model=model,
    data_yaml="data.yaml",
    epochs=30
)

# Plot training history
plot_training_history()
```

#### D. Evaluate Model

```python
# Evaluate performance
eval_results = evaluate_model(
    model=trained_model,
    data_yaml="data.yaml"
)
```

#### E. Process Video

```python
# Process thermal video
detection_counts = process_video(
    video_path="path/to/thermal_video.mp4",
    model=trained_model,
    conf=0.6
)

# Visualize detections over time
plot_video_detections(detection_counts)
```

#### F. Benchmark Model

```python
# Measure inference speed
benchmark_results = benchmark_model(
    model=trained_model,
    image_path="path/to/test_image.jpg",
    num_runs=100
)
```

#### G. Compare Models

```python
# Compare different YOLO variants
comparison_df = compare_models(
    model_names=['yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt'],
    data_yaml="data.yaml"
)
```

#### H. Export Model

```python
# Export for deployment
export_model(
    model=trained_model,
    formats=['onnx', 'torchscript', 'tflite']
)
```

---

## 📂 Project Structure

```
thermal-human-detection/
├── thermal_detection.ipynb      # Main Jupyter notebook
├── data.yaml                    # Dataset configuration
├── outputs/                     # Generated outputs
│   ├── images/                  # Annotated images
│   ├── videos/                  # Processed videos
│   └── plots/                   # Training/evaluation plots
├── runs/                        # YOLO training outputs
│   └── detect/
│       └── train/
│           ├── weights/         # Model checkpoints
│           │   ├── best.pt      # Best model
│           │   └── last.pt      # Latest model
│           └── results.csv      # Training metrics
├── thermal_dataset/             # Your dataset
│   ├── train/
│   ├── valid/
│   └── test/
└── README.md                    # This file
```

---

## 📊 Results

### Expected Performance

After training on a thermal dataset with 1000+ images:

| Metric       | Value  |
|--------------|--------|
| mAP@50       | 0.85+  |
| mAP@50-95    | 0.65+  |
| Precision    | 0.82+  |
| Recall       | 0.78+  |
| Inference    | ~15ms  |
| FPS          | 60+    |

### Sample Outputs

**Training Loss Curves:**
- Box Loss decreases over epochs
- Classification Loss stabilizes
- mAP increases progressively

**Detection Examples:**
- Bounding boxes around detected humans
- Confidence scores displayed
- Works in various thermal imaging conditions

---

## 🐛 Troubleshooting

### Common Issues and Solutions

#### 1. CUDA Out of Memory

**Error:** `RuntimeError: CUDA out of memory`

**Solution:**
```python
# Reduce batch size in Config class
Config.BATCH_SIZE = 8  # or even 4

# Or train on CPU
device = 'cpu'
```

#### 2. OpenCV Import Error

**Error:** `ImportError: libGL.so.1: cannot open shared object file`

**Solution (Linux):**
```bash
sudo apt-get update
sudo apt-get install libgl1-mesa-glx
```

**Solution (Windows/Mac):**
```bash
pip uninstall opencv-python
pip install opencv-python-headless
```

#### 3. YOLO Model Download Fails

**Error:** `HTTPError` or connection timeout

**Solution:**
```python
# Manually download model
# Visit: https://github.com/ultralytics/assets/releases
# Download yolov8n.pt and place in project directory

# Load from local file
model = YOLO('path/to/downloaded/yolov8n.pt')
```

#### 4. Jupyter Kernel Crashes

**Solution:**
```bash
# Increase memory limit
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb=512

# Restart kernel: Kernel → Restart Kernel
```

#### 5. Dataset Not Found

**Error:** `FileNotFoundError: Dataset 'data.yaml' not found`

**Solution:**
```python
# Check if data.yaml exists
from pathlib import Path
print(f"data.yaml exists: {Path('data.yaml').exists()}")

# Use absolute path
data_yaml = "/absolute/path/to/data.yaml"
```

#### 6. Slow Training

**Solution:**
```python
# Enable GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Reduce image size
Config.IMG_SIZE = [416, 416]  # Smaller than default [640, 480]

# Reduce workers
workers = 0  # Single worker for debugging
```

---

## 🔧 Advanced Configuration

### Custom Hyperparameters

```python
# Modify Config class
class Config:
    BATCH_SIZE = 32        # Batch size
    EPOCHS = 50            # Training epochs
    IMG_SIZE = [640, 480]  # Input image size
    LEARNING_RATE = 1e-4   # Learning rate
    CONFIDENCE_THRESHOLD = 0.5  # Detection threshold
```

### Data Augmentation

```python
# Add more augmentation
Config.AUGMENTATION = {
    "hsv_h": 0.02,      # More hue variation
    "hsv_s": 0.8,       # More saturation
    "degrees": 15.0,    # More rotation
    "flipud": 0.7,      # More vertical flips
    "mosaic": 1.0,      # Always use mosaic
    "mixup": 0.2,       # More mixup
}
```

### Multi-GPU Training

```python
# Use multiple GPUs
model.train(
    device=[0, 1, 2],  # GPU IDs
    data="data.yaml",
    **hyperparams
)
```

---

## 📚 Additional Resources

### Documentation
- [YOLOv8 Documentation](https://docs.ultralytics.com/)
- [Supervision Library](https://supervision.roboflow.com/)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)

### Tutorials
- [YOLO Training Guide](https://docs.ultralytics.com/modes/train/)
- [Custom Dataset Training](https://docs.ultralytics.com/datasets/)
- [Model Export Guide](https://docs.ultralytics.com/modes/export/)

### Datasets
- [Thermal Human Dataset (Kaggle)](https://www.kaggle.com/)
- [FLIR Thermal Dataset](https://www.flir.com/oem/adas/adas-dataset-form/)
- [KAIST Multispectral Dataset](https://soonminhwang.github.io/rgbt-ped-detection/)

---

## 🎓 For Academic Use

### Citation

If you use this project in your research, please cite:

```bibtex
@software{thermal_human_detection,
  author = {Md. Abu Said Shabib},
  title = {Thermal Human Detection using YOLOv8},
  year = {2025},
  url = {https://github.com/abusaidshabib/Thermal-Object-Detection.git}
}
```

### Report Structure

For your academic report, include:

1. **Introduction**: Problem statement and objectives
2. **Methodology**: YOLOv8 architecture, training process
3. **Dataset**: Description, preprocessing, augmentation
4. **Results**: Performance metrics, visualizations
5. **Analysis**: Model comparison, benchmark results
6. **Conclusion**: Findings and future work

---

## 🙏 Acknowledgments

- [Ultralytics](https://ultralytics.com/) for YOLOv8
- [Roboflow](https://roboflow.com/) for Supervision library
- [HuggingFace](https://huggingface.co/) for model hosting
- Thermal imaging community for datasets

---
⭐ If you find this project helpful, please consider giving it a star!

