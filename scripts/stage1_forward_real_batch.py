"""Run one real OpenI batch through MedCLIP.

This Stage 1 exercise connects dataset outputs to model embeddings:
image tensor + tokenized report -> shared embedding space -> similarity matrix.
"""

import argparse
from pathlib import Path
import sys

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dataset import OpenIDataset  # noqa: E402
from model import MedCLIP  # noqa: E402


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--indiana-dir", default=None,
                        help="Override cfg data.indiana_dir when data lives outside the repo.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    if args.indiana_dir:
        cfg["data"]["indiana_dir"] = args.indiana_dir

    device = get_device()
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["text_encoder"])
    data_cfg = cfg["data"]
    dataset = OpenIDataset(
        data_cfg["indiana_dir"],
        tokenizer,
        split=args.split,
        image_size=data_cfg["image_size"],
        image_normalization=data_cfg.get("image_normalization", "imagenet"),
        split_dir=data_cfg.get("split_dir"),
        split_ratios=(
            data_cfg.get("train_split", 0.8),
            data_cfg.get("val_split", 0.1),
            data_cfg.get("test_split", 0.1),
        ),
        split_seed=data_cfg.get("split_seed", 42),
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    batch = next(iter(loader))

    model = MedCLIP(
        embed_dim=cfg["model"]["embed_dim"],
        freeze_image_layers=cfg["model"]["freeze_image_layers"],
        temperature=cfg["training"]["temperature"],
    ).to(device)
    if args.checkpoint:
        if not Path(args.checkpoint).exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {args.checkpoint}\n"
                "Train one with `python3 src/train.py --config configs/base.yaml`, "
                "or place a checkpoint at the requested path."
            )
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
        print(f"Loaded checkpoint: {args.checkpoint}")

    model.eval()
    images = batch["image"].to(device)
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)

    with torch.no_grad():
        image_embeddings = model.encode_image(images)
        text_embeddings = model.encode_text(input_ids, attention_mask)
        similarity = image_embeddings @ text_embeddings.T
        logits_per_image, logits_per_text = model(images, input_ids, attention_mask)
        labels = torch.arange(logits_per_image.size(0), device=device)
        image_to_text_loss = F.cross_entropy(logits_per_image, labels)
        text_to_image_loss = F.cross_entropy(logits_per_text, labels)
        symmetric_loss = (image_to_text_loss + text_to_image_loss) / 2
        image_to_text_predictions = logits_per_image.argmax(dim=1)
        text_to_image_predictions = logits_per_text.argmax(dim=1)

    print()
    print("Input shapes")
    print("image:          ", tuple(images.shape))
    print("input_ids:      ", tuple(input_ids.shape))
    print("attention_mask: ", tuple(attention_mask.shape))

    print()
    print("Embedding shapes")
    print("image_embeddings:", tuple(image_embeddings.shape))
    print("text_embeddings: ", tuple(text_embeddings.shape))

    print()
    print("Similarity matrix: rows are images, columns are reports")
    print(similarity.cpu().round(decimals=4))
    print("similarity shape:", tuple(similarity.shape))

    print()
    print("Scaled logits from model.forward")
    print("logits_per_image shape:", tuple(logits_per_image.shape))
    print("logits_per_text shape: ", tuple(logits_per_text.shape))

    print()
    print("InfoNCE loss view")
    print("labels:", labels.cpu().tolist())
    print(f"image-to-text loss: {image_to_text_loss.item():.4f}")
    print(f"text-to-image loss: {text_to_image_loss.item():.4f}")
    print(f"symmetric loss:     {symmetric_loss.item():.4f}")
    print("image-to-text predictions:", image_to_text_predictions.cpu().tolist())
    print("text-to-image predictions:", text_to_image_predictions.cpu().tolist())

    print()
    print("Diagonal interpretation")
    for i in range(args.batch_size):
        print(f"image_{i} is paired with report_{i}")


if __name__ == "__main__":
    main()
