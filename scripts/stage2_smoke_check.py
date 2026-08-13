"""Stage 2 smoke check for the Stage 1 multimodal pipeline."""

import argparse
from pathlib import Path
import sys

import yaml
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dataset import OpenIDataset  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--indiana-dir", default=None,
                        help="Override cfg data.indiana_dir when data lives outside the repo.")
    parser.add_argument("--skip-tokenizer", action="store_true",
                        help="Only verify files/config; do not load Hugging Face tokenizer.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    if args.indiana_dir:
        cfg["data"]["indiana_dir"] = args.indiana_dir

    data_cfg = cfg["data"]
    root = Path(data_cfg["indiana_dir"])
    required = [
        root / "indiana_reports.csv",
        root / "indiana_projections.csv",
        root / "images" / "images_normalized",
    ]

    print(f"Config: {args.config}")
    print(f"OpenI root: {root}")
    for path in required:
        if not path.exists():
            raise FileNotFoundError(f"Missing required OpenI path: {path}")
        print(f"OK: {path}")

    split_dir = Path(data_cfg.get("split_dir", "splits"))
    for split in ["train", "val", "test"]:
        split_path = split_dir / f"openi_{split}_uids.txt"
        status = "found" if split_path.exists() else "not found; deterministic fallback will be used"
        print(f"Split file {split_path}: {status}")

    if args.skip_tokenizer:
        print("Skipping tokenizer/dataloader check.")
        return

    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["text_encoder"])
    dataset = OpenIDataset(
        data_cfg["indiana_dir"],
        tokenizer,
        split="train",
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
    batch = next(iter(DataLoader(dataset, batch_size=2, shuffle=False)))
    print(f"OpenI train samples: {len(dataset)}")
    print("Batch image shape:", tuple(batch["image"].shape))
    print("Batch input_ids shape:", tuple(batch["input_ids"].shape))
    print("Stage 2 smoke check passed.")


if __name__ == "__main__":
    main()
