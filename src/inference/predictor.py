"""Image prediction and inference utilities."""

import logging
from pathlib import Path
from typing import Dict, List, Tuple
import torch
import torch.nn as nn

from utility import preprocess_image

logger = logging.getLogger(__name__)


def predict(
    image_path: str,
    model: nn.Module,
    idx_to_class: Dict[int, str],
    device: torch.device,
    topk: int = 5,
) -> Tuple[List[float], List[str]]:
    """
    Predict the top-K most likely classes for an input image.

    Args:
        image_path: Path to the image file to classify.
        model: Trained PyTorch model for inference.
        idx_to_class: Dictionary mapping class indices to class labels.
        device: The torch device for inference (cuda or cpu).
        topk: Number of top predictions to return. Must be positive.
            Default: 5.

    Returns:
        Tuple[List[float], List[str]]: A tuple containing:
            - List of top-K probabilities (floats between 0 and 1)
            - List of corresponding class labels (strings)

    Raises:
        ValueError: If topk is not positive or image_path is invalid.
        FileNotFoundError: If the image file does not exist.
        RuntimeError: If prediction fails.

    Example:
        >>> probs, labels = predict('flower.jpg', model, idx_to_class, device, topk=3)
        >>> for prob, label in zip(probs, labels):
        ...     print(f"{label}: {prob:.2%}")
    """
    # Input validation
    if topk <= 0:
        raise ValueError(f"topk must be positive, got {topk}")

    if not Path(image_path).exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    try:
        model.eval()
        model.to(device)

        with torch.inference_mode():
            # Preprocess and prepare image tensor
            image = preprocess_image(image_path)
            tensor = torch.from_numpy(image).unsqueeze(0).float().to(device)

            # Forward pass
            logps = model(tensor)
            ps = torch.exp(logps)

            # Get top-K predictions (clamp to available classes)
            k = min(topk, ps.shape[1])
            top_probs, top_indices = ps.topk(k, dim=1)

            # Convert to lists with explicit type casting
            top_probs = [float(v) for v in top_probs.cpu().squeeze().tolist()]
            top_indices = [int(v) for v in top_indices.cpu().squeeze().tolist()]

            # Handle single prediction case
            if not isinstance(top_probs, list):
                top_probs = top_probs
                top_indices = top_indices

            # Map indices to class labels
            top_labels = [idx_to_class[idx] for idx in top_indices]

        return top_probs, top_labels

    except Exception as e:
        logger.error(f"Prediction failed for {image_path}: {str(e)}")
        raise RuntimeError(f"Prediction failed: {str(e)}") from e
