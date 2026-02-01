import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from PIL import Image
import json

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DataConfig:
    """Configuration for dataset loading and preprocessing."""

    BATCH_SIZE: int = 32
    IMG_SIZE: int = 224
    RESIZE_SIZE: int = 255
    IMAGENET_MEAN: list = field(default_factory=lambda: [0.485, 0.456, 0.406])
    IMAGENET_STD: list = field(default_factory=lambda: [0.229, 0.224, 0.225])
    NUM_WORKERS: int = 0
    PHASES: tuple = field(default_factory=lambda: ("train", "valid", "test"))


def dataset_load(
    data_dir: str,
    batch_size: Optional[int] = None,
) -> tuple[Dict[str, DataLoader], Dict[str, datasets.ImageFolder]]:
    """
    Load image datasets and create PyTorch DataLoaders.

    Args:
        data_dir: Path to the data directory (parent of 'flowers' folder).
        batch_size: Override default batch size. Defaults to DataConfig.BATCH_SIZE.

    Returns:
        Dictionary containing DataLoaders for 'train', 'valid', and 'test' phases.

    Raises:
        FileNotFoundError: If data directories don't exist.
        ValueError: If data loading fails.
    """
    config = DataConfig()
    try:
        batch_size = batch_size or config.BATCH_SIZE

        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")

        base_path = Path(data_dir)

        # Validate paths exist
        for phase in config.PHASES:
            phase_path = base_path / phase
            if not phase_path.exists():
                raise FileNotFoundError(f"Data directory not found: {phase_path}")

        logger.info(f"Loading dataset from {base_path}...")

        # Define transformations
        train_transform = transforms.Compose(
            [
                transforms.Resize(config.RESIZE_SIZE),
                transforms.RandomResizedCrop(config.IMG_SIZE),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(config.IMAGENET_MEAN, config.IMAGENET_STD),
            ]
        )

        eval_transform = transforms.Compose(
            [
                transforms.Resize(config.RESIZE_SIZE),
                transforms.CenterCrop(config.IMG_SIZE),
                transforms.ToTensor(),
                transforms.Normalize(config.IMAGENET_MEAN, config.IMAGENET_STD),
            ]
        )

        # Load datasets
        image_datasets = {
            "train": datasets.ImageFolder(
                str(base_path / "train"), transform=train_transform
            ),
            "valid": datasets.ImageFolder(
                str(base_path / "valid"), transform=eval_transform
            ),
            "test": datasets.ImageFolder(
                str(base_path / "test"), transform=eval_transform
            ),
        }

        # Create dataloaders
        dataloaders = {
            "train": DataLoader(
                image_datasets["train"],
                batch_size=batch_size,
                shuffle=True,
                num_workers=config.NUM_WORKERS,
            ),
            "valid": DataLoader(
                image_datasets["valid"],
                batch_size=batch_size,
                shuffle=False,
                num_workers=config.NUM_WORKERS,
            ),
            "test": DataLoader(
                image_datasets["test"],
                batch_size=batch_size,
                shuffle=False,
                num_workers=config.NUM_WORKERS,
            ),
        }

        # Log dataset statistics
        dataset_sizes = {phase: len(image_datasets[phase]) for phase in config.PHASES}
        logger.info(
            f"Dataset loaded - Train: {dataset_sizes['train']}, Valid: {dataset_sizes['valid']}, Test: {dataset_sizes['test']}"
        )

        return dataloaders, image_datasets

    except FileNotFoundError as e:
        logger.error(f"Data loading error: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error while loading dataset: {e}")
        raise


def label_mapping(mapping_file: str) -> Dict[int, str]:
    """
    Load label mapping from a JSON file.

    Args:
        mapping_file: Path to the JSON file containing label mappings.

    Returns:
        Dictionary mapping class indices to human-readable labels.

    Raises:
        FileNotFoundError: If the mapping file does not exist.
        ValueError: If JSON decoding fails.
    """

    try:
        with open(mapping_file, "r") as f:
            class_to_name = json.load(f)
        return {int(k): v for k, v in class_to_name.items()}
    except FileNotFoundError as e:
        logger.error(f"Mapping file not found: {e}")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON from mapping file: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error while loading label mapping: {e}")
        raise


def preprocess_image(img_path: str) -> np.ndarray:
    # Open image
    image = Image.open(img_path)

    # Resize
    if image.width > image.height:
        new_height = 256
        new_width = int(image.width * (256 / image.height))
    else:
        new_width = 256
        new_height = int(image.height * (256 / image.width))
    image = image.resize((new_width, new_height))

    # Crop
    left = (image.width - 224) / 2
    top = (image.height - 224) / 2
    right = left + 224
    bottom = top + 224
    image = image.crop((left, top, right, bottom))

    # Normalize
    np_image = np.array(image) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - mean) / std

    # Reorder dimensions
    np_image = np_image.transpose((2, 0, 1))

    return np_image


def get_device(gpu: bool) -> torch.device:
    """Get the available device (GPU if available, else CPU)."""
    return torch.device("cuda" if torch.cuda.is_available() and gpu else "cpu")
