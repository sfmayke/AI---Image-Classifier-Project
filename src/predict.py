import argparse
import logging

from model import load_checkpoint, predict
from utility import get_device, label_mapping

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Model Prediction Script")
    parser.add_argument(
        "--img-path",
        type=str,
        help="Path to the image for prediction",
    )
    parser.add_argument(
        "--checkpoint-path", type=str, help="Path to the model checkpoint"
    )
    parser.add_argument(
        "--top-k", type=int, default=3, help="Number of top predictions to return"
    )
    parser.add_argument(
        "--gpu", type=bool, default=False, help="Use GPU for prediction if available"
    )
    parser.add_argument(
        "--labels-map-path",
        type=str,
        default="cat_to_name.json",
        help="Path to the labels mapping JSON file",
    )

    args = parser.parse_args()

    # check if --img-path and --checkpoint-path are provided
    if not args.img_path or not args.checkpoint_path:
        parser.error("Both --img-path and --checkpoint-path are required.")

    model = load_checkpoint(args.checkpoint_path)
    device = get_device(args.gpu)

    idx_to_class = {v: k for k, v in model.class_to_idx.items()}
    predictions = predict(
        args.img_path, model, topk=args.top_k, device=device, idx_to_class=idx_to_class
    )

    labels_map = label_mapping(args.labels_map_path)
    for ps, label_idx in zip(*predictions):
        cat = labels_map.get(int(label_idx), "Unknown")
        logger.info(f"Prediction: {cat} with probability {ps}")


if __name__ == "__main__":
    main()
