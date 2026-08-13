"""Inspect one real OpenI batch for Stage 1 multimodal learning.

This script shows how a real chest X-ray/report pair becomes the model inputs:
image tensor, token ids, and attention mask.
"""

import argparse
from pathlib import Path

import yaml
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dataset import OpenIDataset  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--indiana-dir", default=None,
                        help="Override cfg data.indiana_dir when data lives outside the repo.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    if args.indiana_dir:
        cfg["data"]["indiana_dir"] = args.indiana_dir

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

    print(f"Dataset root: {cfg['data']['indiana_dir']}")
    print(f"Split: {args.split}")
    print(f"Dataset size: {len(dataset)}")
    print()

    print("Batch keys:", list(batch.keys()))
    print("image shape:         ", tuple(batch["image"].shape))
    print("input_ids shape:     ", tuple(batch["input_ids"].shape))
    print("attention_mask shape:", tuple(batch["attention_mask"].shape))
    print()

    sample = dataset.samples[args.sample_index]
    print(f"Raw sample #{args.sample_index}")
    print("uid:       ", sample["uid"])
    print("image_path:", sample["image_path"])
    print("caption:")
    print(sample["caption"][:1000])
    print()

    token_ids = batch["input_ids"][0]
    attention_mask = batch["attention_mask"][0]
    active_token_count = int(attention_mask.sum().item())
    decoded_text = tokenizer.decode(token_ids, skip_special_tokens=True)

    print("First batch item token view")
    print("first 30 token ids:", token_ids[:30].tolist())
    print("first 30 mask vals:", attention_mask[:30].tolist())
    print("active token count:", active_token_count)
    print("decoded text:")
    print(decoded_text[:1000])


if __name__ == "__main__":
    main()
