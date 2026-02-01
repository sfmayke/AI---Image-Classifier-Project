# AI - Image Classifier Project

## Overview
This project trains a deep learning image classifier to recognize 102 flower categories using PyTorch and a DenseNet121 backbone. It includes scripts for training a model, saving checkpoints, and running predictions on new images.

## Project Structure
- assets/ – project assets
- data/flowers/ – dataset organized into train/valid/test
- notebooks/ – exploratory notebook(s)
- src/ – source code
  - train.py – training entry point
  - predict.py – prediction entry point
  - model.py – model creation/training utilities
  - utility.py – data loading, device helpers, label mapping
  - cat_to_name.json – class index to flower name mapping

## Setup
- Python 3.8+
- PyTorch + torchvision

Install dependencies (example):
- pip install torch torchvision

## Training
From the src/ directory:
- python train.py --data-dir ../data/flowers --epochs 5 --learning-rate 0.001

Key arguments:
- --data-dir: path to data/flowers
- --batch-size: batch size
- --epochs: number of epochs
- --learning-rate: learning rate
- --labels-dir: label map JSON
- --gpu: use GPU (True/False)

## Prediction
From the src/ directory:
- python predict.py --img-path ../data/flowers/valid/1/image_06765.jpg --checkpoint-path ./trained_model.pth

Key arguments:
- --img-path: path to image
- --checkpoint-path: saved model checkpoint
- --top-k: number of predictions
- --labels-map-path: label map JSON

## Notes
- The dataset should be in data/flowers with train/valid/test subfolders.
- Update paths as needed for your environment.

## License
For educational use.