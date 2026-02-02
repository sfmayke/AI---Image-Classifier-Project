"""Model checkpoint saving and loading utilities."""

import logging
from pathlib import Path
from typing import Dict, cast
import torch
import torch.nn as nn
from torchvision.models import densenet121

logger = logging.getLogger(__name__)


def save_model(
    model: nn.Module,
    path: str,
    class_to_idx: Dict[str, int],
) -> None:
    """
    Save the model to a checkpoint file.

    Args:
        model: The PyTorch model to save.
        path: File path to save the checkpoint.
        class_to_idx: Mapping from class labels to indices.

    Raises:
        IOError: If saving the model fails.

    Example:
        >>> save_model(model, 'trained_model.pth', dataset.class_to_idx)
    """
    checkpoint = {
        "state_dict": model.state_dict(),
        "class_to_idx": class_to_idx,
        "classifier": model.classifier,
    }
    try:
        torch.save(checkpoint, path)
        logger.info(f"Model saved to {path}")
    except IOError as e:
        logger.error(f"Failed to save model to {path}: {str(e)}")
        raise IOError(f"Saving model failed: {str(e)}") from e


def load_checkpoint(filepath: str) -> nn.Module:
    """
    Load a model checkpoint from a file.

    Args:
        filepath: Path to the checkpoint file.

    Returns:
        nn.Module: The loaded PyTorch model with attached class_to_idx attribute.

    Raises:
        FileNotFoundError: If the checkpoint file does not exist.
        KeyError: If checkpoint is missing required keys.
        RuntimeError: If loading the checkpoint fails.

    Example:
        >>> model = load_checkpoint('trained_model.pth')
        >>> print(model.class_to_idx)
    """
    # Validate file exists
    if not Path(filepath).exists():
        raise FileNotFoundError(f"Checkpoint file not found: {filepath}")

    try:
        logger.info(f"Loading checkpoint from {filepath}")

        # Load checkpoint
        checkpoint = torch.load(filepath, weights_only=False)

        # Validate checkpoint structure
        required_keys = {"state_dict", "class_to_idx", "classifier"}
        missing_keys = required_keys - checkpoint.keys()
        if missing_keys:
            raise KeyError(f"Checkpoint missing required keys: {missing_keys}")

        # Create base model
        model = densenet121(weights="IMAGENET1K_V1")

        # Freeze feature extractor
        for param in model.parameters():
            param.requires_grad = False

        # Restore classifier and weights
        model.classifier = cast(nn.Linear, checkpoint["classifier"])
        model.load_state_dict(checkpoint["state_dict"])

        # Attach class mapping
        model.class_to_idx = checkpoint["class_to_idx"]

        logger.info("Model loaded successfully.")

        return model

    except Exception as e:
        logger.error(f"Failed to load checkpoint from {filepath}: {str(e)}")
        raise RuntimeError(f"Checkpoint loading failed: {str(e)}") from e
