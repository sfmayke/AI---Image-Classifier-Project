import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import densenet121

from utility import preprocess_image

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")


def create_model(device: torch.device) -> nn.Module:
    """
    Create and return a DenseNet121 model modified for 102 output classes.

    Returns:
        model: A PyTorch model instance.
    """
    model = densenet121(pretrained=True)

    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False

    classifier = nn.Sequential(
        nn.Linear(1024, 512),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(512, 102),
        nn.LogSoftmax(dim=1),
    )

    model.classifier = classifier
    model.to(device)
    return model


def train_model(
    model: nn.Module,
    dataloaders: dict,
    device: torch.device,
    learning_rate: float,
    epochs: int,
) -> None:
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    steps = 0
    print_every = 40
    for epoch in range(epochs):
        running_loss = 0
        for inputs, labels in dataloaders["train"]:
            steps += 1
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            logps = model(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                valid_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in dataloaders["valid"]:
                        inputs, labels = inputs.to(device), labels.to(device)
                        logps = model(inputs)
                        batch_loss = criterion(logps, labels)
                        valid_loss += batch_loss.item()

                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                logger.info(
                    f"Epoch {epoch+1}/{epochs}.. "
                    f"Train loss: {running_loss/print_every:.3f}.. "
                    f"Validation loss: {valid_loss/len(dataloaders['valid']):.3f}.. "
                    f"Validation accuracy: {accuracy/len(dataloaders['valid']):.3f}"
                )
                running_loss = 0
                model.train()


def predict(image_path, model, topk, device, idx_to_class):
    """Predict top-K classes for a single image."""

    model.eval()
    with torch.inference_mode():
        image = preprocess_image(image_path)
        tensor = torch.from_numpy(image).unsqueeze(0).float().to(device)
        model.to(device)

        logps = model(tensor)
        ps = torch.exp(logps)

        # Clamp topk to number of classes in output
        k = min(topk, ps.shape[1])
        top_p, top_class = ps.topk(k, dim=1)

        top_class = top_class.cpu().numpy().squeeze().tolist()
        top_p = top_p.cpu().numpy().squeeze().tolist()

        # Ensure list output for single-item case
        if isinstance(top_class, int):
            top_class = [top_class]
        if isinstance(top_p, float):
            top_p = [top_p]

        top_labels = [idx_to_class[i] for i in top_class]

    return top_p, top_labels


def save_model(
    model: nn.Module,
    path: str,
    class_to_idx: dict,
) -> None:
    checkpoint = {
        "state_dict": model.state_dict(),
        "class_to_idx": class_to_idx,
        "classifier": model.classifier,
    }
    torch.save(checkpoint, path)
    logger.info(f"Model saved to {path}")


def load_checkpoint(filepath: str) -> nn.Module:
    checkpoint = torch.load(filepath)
    model = densenet121(pretrained=True)

    logger.info(checkpoint["state_dict"])

    for param in model.parameters():
        param.requires_grad = False

    model.classifier = checkpoint["classifier"]
    model.load_state_dict(checkpoint["state_dict"])

    model.class_to_idx = checkpoint["class_to_idx"]

    logger.info(f"Model loaded from {filepath}")

    return model
