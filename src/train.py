import argparse
import logging

import torch
from models import create_model, save_model, train_model
from utility import dataset_load, get_device, label_mapping

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Model Train Script")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="../data/flowers",
        help="Directory containing the training data",
    )
    parser.add_argument(
        "--batch-size", type=int, default=32, help="Batch size for training"
    )
    parser.add_argument(
        "--epochs", type=int, default=5, help="Number of training epochs"
    )
    parser.add_argument(
        "--learning-rate", type=float, default=0.003, help="Learning rate"
    )
    parser.add_argument(
        "--labels-dir",
        type=str,
        default="cat_to_name.json",
        help="Path to label mapping file",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default=".",
        help="Directory to save the trained model",
    )
    parser.add_argument("--gpu", type=bool, default=False, help="Use GPU for training")
    args = parser.parse_args()

    device = get_device(args.gpu)
    dataloaders, image_datasets = dataset_load(args.data_dir, args.batch_size)
    model = create_model(device)

    # Train the model
    train_model(
        model,
        dataloaders,
        device,
        args.learning_rate,
        args.epochs,
    )

    # Save the trained model
    save_model(
        model,
        args.save_dir + "/trained_model.pth",
        image_datasets["train"].class_to_idx,
    )

    logger.info("Model training and saving completed.")


if __name__ == "__main__":
    main()
