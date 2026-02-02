# Project Structure

This project has been refactored into a modular architecture for better maintainability and scalability.

## Directory Structure

```
src/
├── models/                   # Model-related modules
│   ├── __init__.py          # Package exports
│   ├── architecture.py      # Model creation and configuration
│   ├── training.py          # Training and validation logic
│   └── checkpoints.py       # Model persistence (save/load)
│
├── inference/               # Inference-related modules
│   ├── __init__.py         # Package exports
│   └── predictor.py        # Prediction and inference logic
│
├── utils/                   # Utility functions (existing)
│   └── utility.py          # Data loading, preprocessing, etc.
│
├── train.py                # CLI entry point for training
├── predict.py              # CLI entry point for prediction
└── model.py                # [DEPRECATED] Legacy monolithic module
```

## Module Responsibilities

### `models/architecture.py`
- **Purpose**: Model architecture creation and configuration
- **Key Function**: `create_model()`
- **Responsibilities**:
  - Create DenseNet121 base model
  - Configure custom classifier
  - Freeze/unfreeze layers
  - Device management

### `models/training.py`
- **Purpose**: Training and validation logic
- **Key Functions**: `train_model()`, `validate_model()`
- **Responsibilities**:
  - Complete training loop
  - Validation monitoring
  - Gradient clipping
  - Training history tracking
  - Metrics logging

### `models/checkpoints.py`
- **Purpose**: Model persistence
- **Key Functions**: `save_model()`, `load_checkpoint()`
- **Responsibilities**:
  - Save model checkpoints
  - Load pre-trained models
  - Checkpoint validation
  - Error handling for I/O operations

### `inference/predictor.py`
- **Purpose**: Image prediction and inference
- **Key Function**: `predict()`
- **Responsibilities**:
  - Image preprocessing
  - Forward pass inference
  - Top-K predictions
  - Class label mapping

## Usage Examples

### Training
```python
from models import create_model, train_model, save_model

# Create model
model = create_model(device, num_classes=102, hidden_units=512)

# Train model
history = train_model(model, dataloaders, device, epochs=10)

# Save model
save_model(model, 'trained_model.pth', class_to_idx)
```

### Inference
```python
from models import load_checkpoint
from inference import predict

# Load model
model = load_checkpoint('trained_model.pth')

# Make prediction
probs, labels = predict('flower.jpg', model, idx_to_class, device, topk=5)
```

## Benefits of This Structure

1. **Separation of Concerns**: Each module has a single, well-defined responsibility
2. **Easier Testing**: Mock and test components independently
3. **Better Reusability**: Import only what you need
4. **Improved Maintainability**: Changes are localized to specific modules
5. **Team Collaboration**: Multiple developers can work on different modules
6. **Clear Dependencies**: Import structure shows module relationships

## Migration Notes

### Old Import Pattern
```python
from model import create_model, train_model, predict, save_model, load_checkpoint
```

### New Import Pattern
```python
from models import create_model, train_model, save_model, load_checkpoint
from inference import predict
```

## Next Steps

1. ✅ Refactored monolithic `model.py` into modular structure
2. ✅ Updated `train.py` and `predict.py` to use new imports
3. ⏭️ (Optional) Add unit tests for each module
4. ⏭️ (Optional) Add type stubs (`.pyi` files) for better IDE support
5. ⏭️ (Optional) Deprecate and remove `model.py` after confirming everything works

## Testing the New Structure

Run your existing commands to verify everything works:

```bash
# Training
python train.py --epochs 3 --learning-rate=0.003 --gpu True

# Prediction
python predict.py --img-path='../data/flowers/valid/1/image_06765.jpg' --checkpoint-path='./trained_model.pth'
```

Both should work identically to before!
