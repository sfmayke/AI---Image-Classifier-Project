"""Model training and validation logic."""

import logging
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


def train_model(
    model: nn.Module,
    dataloaders: Dict[str, DataLoader],
    device: torch.device,
    learning_rate: float = 0.001,
    epochs: int = 5,
    criterion: Optional[nn.Module] = None,
    optimizer: Optional[optim.Optimizer] = None,
    print_every: int = 40,
) -> Dict[str, List[float]]:
    """
    Train a PyTorch model with comprehensive logging, validation, and optional features.

    This function implements a complete training loop with validation monitoring
    and gradient clipping.

    Args:
        model: The PyTorch model to train. Must have trainable parameters.
        dataloaders: Dictionary containing 'train' and 'valid' DataLoader objects.
            Keys must be 'train' and 'valid'.
        device: The torch device for training (cuda or cpu).
        learning_rate: Initial learning rate for the optimizer.
            Must be positive. Default: 0.001.
        epochs: Number of training epochs. Must be positive. Default: 5.
        criterion: Loss function. If None, uses nn.NLLLoss(). Default: None.
        optimizer: Optimizer for training. If None, creates Adam optimizer
            for classifier parameters. Default: None.
        print_every: Frequency of validation checks (in training steps).
            Must be positive. Default: 40.

    Returns:
        Dict[str, List[float]]: Training history containing:
            - 'train_losses': List of training losses at each validation check
            - 'valid_losses': List of validation losses at each validation check
            - 'valid_accuracies': List of validation accuracies at each validation check
            - 'learning_rates': List of learning rates at each validation check

    Raises:
        ValueError: If dataloaders missing required keys, invalid parameters,
            or model has no trainable parameters.
        RuntimeError: If training fails due to CUDA errors or other issues.

    Example:
        >>> from models.architecture import create_model
        >>> model = create_model(device)
        >>> history = train_model(
        ...     model, dataloaders, device,
        ...     learning_rate=0.001, epochs=10
        ... )
        >>> print(f"Best validation accuracy: {max(history['valid_accuracies']):.3f}")
    """
    # Input validation
    if not isinstance(dataloaders, dict):
        raise ValueError(f"dataloaders must be a dict, got {type(dataloaders)}")

    required_keys = {"train", "valid"}
    if not required_keys.issubset(dataloaders.keys()):
        raise ValueError(
            f"dataloaders must contain keys {required_keys}, got {dataloaders.keys()}"
        )

    if not isinstance(device, torch.device):
        raise ValueError(f"device must be a torch.device, got {type(device)}")

    if learning_rate <= 0:
        raise ValueError(f"learning_rate must be positive, got {learning_rate}")

    if epochs <= 0:
        raise ValueError(f"epochs must be positive, got {epochs}")

    if print_every <= 0:
        raise ValueError(f"print_every must be positive, got {print_every}")

    # Check for trainable parameters
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params:
        raise ValueError("Model has no trainable parameters")

    # Initialize criterion and optimizer if not provided
    if criterion is None:
        criterion = nn.NLLLoss()
        logger.info("Using default criterion: NLLLoss")

    if optimizer is None:
        optimizer = optim.Adam(trainable_params, lr=learning_rate)
        logger.info(f"Using default optimizer: Adam with lr={learning_rate}")

    # Training state
    history = {
        "train_losses": [],
        "valid_losses": [],
        "valid_accuracies": [],
        "learning_rates": [],
    }

    total_steps = 0

    try:
        logger.info(
            f"Starting training: {epochs} epochs, "
            f"lr={learning_rate}, print_every={print_every}"
        )
        logger.info(
            f"Training batches: {len(dataloaders['train'])}, "
            f"Validation batches: {len(dataloaders['valid'])}"
        )

        model.train()

        for epoch in range(epochs):
            epoch_loss = 0.0
            running_loss = 0.0

            for batch_idx, (inputs, labels) in enumerate(dataloaders["train"]):
                total_steps += 1

                # Move data to device
                inputs, labels = inputs.to(device), labels.to(device)

                # Zero gradients
                optimizer.zero_grad()

                # Forward pass
                logps = model(inputs)
                loss = criterion(logps, labels)

                # Backward pass with gradient clipping
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                # Track loss
                batch_loss = loss.item()
                running_loss += batch_loss
                epoch_loss += batch_loss

                # Validation step
                if total_steps % print_every == 0:
                    valid_loss, valid_accuracy = validate_model(
                        model, dataloaders["valid"], criterion, device
                    )

                    # Get current learning rate
                    current_lr = optimizer.param_groups[0]["lr"]

                    # Log metrics
                    train_loss_avg = running_loss / print_every
                    history["train_losses"].append(train_loss_avg)
                    history["valid_losses"].append(valid_loss)
                    history["valid_accuracies"].append(valid_accuracy)
                    history["learning_rates"].append(current_lr)

                    logger.info(
                        f"Epoch {epoch+1}/{epochs} | Step {total_steps} | "
                        f"Train Loss: {train_loss_avg:.4f} | "
                        f"Valid Loss: {valid_loss:.4f} | "
                        f"Valid Acc: {valid_accuracy:.4f} | "
                        f"LR: {current_lr:.6f}"
                    )

                    running_loss = 0.0
                    model.train()

            # Epoch complete
            avg_epoch_loss = epoch_loss / len(dataloaders["train"])
            logger.info(
                f"Epoch {epoch+1}/{epochs} complete. Avg train loss: {avg_epoch_loss:.4f}"
            )

        logger.info("Training complete!")
        return history

    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise RuntimeError(f"Training failed: {str(e)}") from e


def validate_model(
    model: nn.Module, dataloader: DataLoader, criterion: nn.Module, device: torch.device
) -> Tuple[float, float]:
    """
    Validate the model on a validation dataset.

    Args:
        model: The model to validate.
        dataloader: DataLoader for validation data.
        criterion: Loss function.
        device: Device to run validation on.

    Returns:
        Tuple[float, float]: (average validation loss, accuracy)
    """
    model.eval()
    valid_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            logps = model(inputs)
            batch_loss = criterion(logps, labels)
            valid_loss += batch_loss.item()

            # Calculate accuracy
            ps = torch.exp(logps)
            top_class = ps.argmax(dim=1)
            correct += (top_class == labels).sum().item()
            total += labels.size(0)

    avg_loss = valid_loss / len(dataloader)
    accuracy = correct / total

    return avg_loss, accuracy
