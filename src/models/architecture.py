"""Model architecture creation and configuration."""

import logging
from typing import cast
import torch
import torch.nn as nn
from torchvision.models import densenet121

logger = logging.getLogger(__name__)


def create_model(
    device: torch.device,
    num_classes: int = 102,
    hidden_units: int = 512,
    dropout_rate: float = 0.1,
    freeze_features: bool = True,
    pretrained: bool = True,
) -> nn.Module:
    """
    Create a DenseNet121 model with a custom classifier for transfer learning.

    This function creates a pre-trained DenseNet121 model and replaces its classifier
    with a custom fully-connected network. By default, the feature extraction layers
    are frozen to prevent backpropagation through them during training.

    Args:
        device: The torch device to place the model on (cuda or cpu).
        num_classes: Number of output classes for the classifier. Must be positive.
            Default: 102 (for flower classification).
        hidden_units: Number of neurons in the hidden layer of the classifier.
            Must be positive and typically between 256 and 2048.
            Default: 512.
        dropout_rate: Dropout probability for regularization. Must be in range [0, 1).
            Higher values provide more regularization but may underfit.
            Default: 0.1.
        freeze_features: Whether to freeze the feature extractor weights.
            Set to False to fine-tune the entire network.
            Default: True.
        pretrained: Whether to load ImageNet pre-trained weights.
            Default: True.

    Returns:
        nn.Module: A configured DenseNet121 model ready for training or inference.

    Raises:
        ValueError: If any parameter is invalid (e.g., negative num_classes,
            dropout_rate outside [0, 1) range).
        RuntimeError: If model creation or device transfer fails.

    Example:
        >>> device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        >>> model = create_model(device, num_classes=102, hidden_units=512)
        >>> print(model.classifier)
    """
    # Input validation
    if not isinstance(device, torch.device):
        raise ValueError(f"device must be a torch.device, got {type(device)}")

    if num_classes <= 0:
        raise ValueError(f"num_classes must be positive, got {num_classes}")

    if hidden_units <= 0:
        raise ValueError(f"hidden_units must be positive, got {hidden_units}")

    if not (0 <= dropout_rate < 1):
        raise ValueError(f"dropout_rate must be in [0, 1), got {dropout_rate}")

    try:
        logger.info(
            f"Creating DenseNet121 model: "
            f"num_classes={num_classes}, hidden_units={hidden_units}, "
            f"dropout_rate={dropout_rate}, pretrained={pretrained}, "
            f"freeze_features={freeze_features}"
        )

        # Load base model with weights parameter (updated from deprecated pretrained)
        model = densenet121(weights="IMAGENET1K_V1" if pretrained else None)

        # Get input features from the original classifier
        num_features = model.classifier.in_features
        logger.info(
            f"DenseNet121 base model loaded. Classifier input features: {num_features}"
        )

        # Freeze feature extraction layers if specified
        if freeze_features:
            for param in model.parameters():
                param.requires_grad = False
            logger.info("Feature extractor weights frozen")

        # Create custom classifier with batch normalization for better training stability
        classifier = nn.Sequential(
            nn.Linear(num_features, hidden_units),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(hidden_units),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_units, num_classes),
            nn.LogSoftmax(dim=1),
        )

        # Replace the classifier
        model.classifier = cast(nn.Linear, classifier)
        logger.info(
            f"Custom classifier created and attached: {num_features} -> {hidden_units} -> {num_classes}"
        )

        # Move model to device
        model.to(device)
        logger.info(f"Model transferred to device: {device}")

        return model

    except Exception as e:
        logger.error(f"Failed to create model: {str(e)}")
        raise RuntimeError(f"Model creation failed: {str(e)}") from e
