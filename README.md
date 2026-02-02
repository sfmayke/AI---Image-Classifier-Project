# AI - Image Classifier Project

## Overview
This project trains a deep learning image classifier to recognize 102 flower categories using PyTorch and a DenseNet121 backbone. It includes scripts for training a model, saving checkpoints, and running predictions on new images.

## Project Structure
```
├── assets/                  # Project assets
├── data/flowers/            # Dataset (train/valid/test)
├── notebooks/               # Exploratory notebooks
└── src/                     # Source code
    ├── models/              # Model-related modules
    │   ├── architecture.py  # Model creation and configuration
    │   ├── training.py      # Training and validation logic
    │   └── checkpoints.py   # Model persistence (save/load)
    ├── inference/           # Inference modules
    │   └── predictor.py     # Prediction and inference
    ├── train.py             # CLI training entry point
    ├── predict.py           # CLI prediction entry point
    ├── utility.py           # Data loading, preprocessing
    └── cat_to_name.json     # Class index to flower name mapping
```

### Module Responsibilities
- **models/architecture.py**: DenseNet121 model creation with custom classifier
- **models/training.py**: Complete training loop with validation and metrics
- **models/checkpoints.py**: Save/load model checkpoints
- **inference/predictor.py**: Image prediction with top-K class probabilities
- **utility.py**: Dataset loading, device management, label mapping

## Setup

**Requirements:**
- Python 3.8+
- PyTorch 2.0+
- torchvision

**Installation:**
```bash
# Install PyTorch (visit pytorch.org for your specific configuration)
pip install torch torchvision

# Or with CUDA support
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

## Architecture Details

**Model:** DenseNet121 with custom classifier
- **Base:** Pre-trained DenseNet121 on ImageNet
- **Custom Classifier:**
  - Linear (1024 → 512)
  - ReLU + BatchNorm1d
  - Dropout (0.1)
  - Linear (512 → 102)
  - LogSoftmax
- **Training:** Transfer learning with frozen feature extractor
- **Loss:** Negative Log Likelihood Loss
- **Optimizer:** Adam with configurable learning rate
- **Features:** Gradient clipping, validation monitoring, training history tracking

## Training
From the `src/` directory:
```bash
python train.py --data-dir ../data/flowers --epochs 5 --learning-rate 0.001 --gpu True
```

**Key Arguments:**
- `--data-dir`: Path to data/flowers directory (default: `../data/flowers`)
- `--batch-size`: Training batch size (default: `32`)
- `--epochs`: Number of training epochs (default: `5`)
- `--learning-rate`: Learning rate for optimizer (default: `0.003`)
- `--labels-dir`: Path to label mapping JSON (default: `cat_to_name.json`)
- `--save-dir`: Directory to save trained model (default: `.`)
- `--gpu`: Use GPU if available (default: `False`)

**Example:**
```bash
python train.py --epochs 10 --learning-rate 0.003 --gpu True
```

The trained model will be saved as `trained_model.pth` in the specified save directory.

## Prediction
From the `src/` directory:
```bash
python predict.py --img-path ../data/flowers/valid/1/image_06765.jpg --checkpoint-path ./trained_model.pth
```

**Key Arguments:**
- `--img-path`: Path to image file for classification (required)
- `--checkpoint-path`: Path to saved model checkpoint (required)
- `--top-k`: Number of top predictions to return (default: `3`)
- `--gpu`: Use GPU if available (default: `False`)
- `--labels-map-path`: Path to label mapping JSON (default: `cat_to_name.json`)

**Example:**
```bash
python predict.py \
  --img-path ../data/flowers/test/1/image_06743.jpg \
  --checkpoint-path ./trained_model.pth \
  --top-k 5 \
  --gpu True
```

**Output:**
```
Prediction: pink primrose with probability 0.9234
Prediction: hard-leaved pocket orchid with probability 0.0421
Prediction: canterbury bells with probability 0.0156
```

## Notes
- Dataset should be organized in `data/flowers/` with `train/`, `valid/`, and `test/` subdirectories
- Each subdirectory contains numbered folders (1-102) representing flower categories
- Images are automatically preprocessed with standard ImageNet normalization
- Model checkpoints include full state dict, classifier architecture, and class mappings
- Supports both CPU and GPU training/inference
- Dataset source: [Oxford 102 Flower Dataset](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html)

## License
For educational use.
