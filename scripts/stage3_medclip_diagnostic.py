"""Evaluate a fine-tuned checkpoint's retrieval and matched-vs-random alignment.

Works for both backbones (`--backbone medclip|biomedclip`); the BioMedCLIP wrapper
brings its own preprocessing and tokenizer.
"""

import argparse
from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dataset import OpenIDataset  # noqa: E402
from train import build_backbone  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--checkpoint", default="checkpoints/best.pt")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--indiana-dir", default=None)
    parser.add_argument("--image-normalization", choices=["imagenet", "clip"], default=None,
                        help="Override cfg data.image_normalization for preprocessing ablations.")
    parser.add_argument("--backbone", choices=["medclip", "biomedclip"], default=None,
                        help="Override cfg model.backbone. BioMedCLIP ignores the normalization flag.")
    parser.add_argument("--out", default="stage3_medclip_diagnostic.md")
    return parser.parse_args()


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def openi_kwargs(cfg: dict) -> dict:
    data_cfg = cfg["data"]
    return {
        "split_dir": data_cfg.get("split_dir"),
        "split_ratios": (
            data_cfg.get("train_split", 0.8),
            data_cfg.get("val_split", 0.1),
            data_cfg.get("test_split", 0.1),
        ),
        "split_seed": data_cfg.get("split_seed", 42),
    }


def recall_at_k(similarity: np.ndarray, k_values: list[int]) -> dict[str, float]:
    n = similarity.shape[0]
    ranks = np.argsort(-similarity, axis=1)
    ground_truth = np.arange(n)
    metrics = {}
    for k in k_values:
        hits = np.any(ranks[:, :k] == ground_truth[:, None], axis=1)
        metrics[f"R@{k}"] = float(hits.mean() * 100)
    metrics["MedR"] = float(np.median(np.argwhere(ranks == ground_truth[:, None])[:, 1] + 1))
    return metrics


def matched_random_stats(similarity: np.ndarray, seed: int = 42) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    n = similarity.shape[0]
    matched = np.diag(similarity)
    random_cols = rng.permutation(n)
    fixed_points = random_cols == np.arange(n)
    if fixed_points.any():
        random_cols[fixed_points] = (random_cols[fixed_points] + 1) % n
    random_scores = similarity[np.arange(n), random_cols]
    gap = float(matched.mean() - random_scores.mean())
    random_std = float(random_scores.std())
    return {
        "matched_mean": float(matched.mean()),
        "matched_std": float(matched.std()),
        "random_mean": float(random_scores.mean()),
        "random_std": random_std,
        "gap": gap,
        # The raw gap is not scale-free: models whose embeddings occupy a narrow
        # cone (BioMedCLIP: random pairs already at ~0.39 cosine) show a small
        # absolute gap even when they rank well. Dividing by the spread of the
        # random scores gives an effect size that is comparable across backbones.
        "gap_normalized": gap / random_std if random_std > 0 else float("nan"),
    }


@torch.no_grad()
def encode_dataset(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    image_embeddings = []
    text_embeddings = []
    for batch in loader:
        image_embeddings.append(model.encode_image(batch["image"].to(device)).cpu())
        text_embeddings.append(model.encode_text(
            batch["input_ids"].to(device),
            batch["attention_mask"].to(device),
        ).cpu())
    return torch.cat(image_embeddings), torch.cat(text_embeddings)


def write_report(
    out_path: Path,
    checkpoint: str,
    split: str,
    sample_count: int,
    i2t: dict[str, float],
    t2i: dict[str, float],
    stats: dict[str, float],
) -> None:
    lines = [
        "# Stage 3 Fine-Tuned MedCLIP Diagnostic",
        "",
        "This run evaluates the project's fine-tuned MedCLIP checkpoint on OpenI retrieval.",
        "",
        f"- Checkpoint: `{checkpoint}`",
        f"- Split: `{split}`",
        f"- Samples: `{sample_count}`",
        "",
        "## Retrieval Metrics",
        "",
        "| Direction | R@1 | R@5 | R@10 | MedR |",
        "|-----------|-----|-----|------|------|",
        (
            f"| Image -> Text | {i2t['R@1']:.2f} | {i2t['R@5']:.2f} | "
            f"{i2t['R@10']:.2f} | {i2t['MedR']:.2f} |"
        ),
        (
            f"| Text -> Image | {t2i['R@1']:.2f} | {t2i['R@5']:.2f} | "
            f"{t2i['R@10']:.2f} | {t2i['MedR']:.2f} |"
        ),
        "",
        "## Matched vs Random Similarity",
        "",
        "| Pair type | Mean | Std |",
        "|-----------|------|-----|",
        f"| Matched image/report | {stats['matched_mean']:.4f} | {stats['matched_std']:.4f} |",
        f"| Random mismatched pair | {stats['random_mean']:.4f} | {stats['random_std']:.4f} |",
        "",
        f"Matched-minus-random gap: `{stats['gap']:.4f}`",
        f"Normalized gap (gap / random std): `{stats['gap_normalized']:.4f}`",
        "",
        "Interpretation: a positive gap means paired X-ray/report examples are closer than random mismatches in the learned shared embedding space.",
        "",
        "Compare the raw gap only within one backbone. Across backbones use the normalized gap or the",
        "ranking metrics: the raw gap scales with how tightly the embeddings are packed, so a model whose",
        "embeddings sit in a narrow cone can rank better while showing a much smaller absolute gap.",
    ]
    out_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    if args.indiana_dir:
        cfg["data"]["indiana_dir"] = args.indiana_dir
    if args.image_normalization:
        cfg["data"]["image_normalization"] = args.image_normalization
    if args.backbone:
        cfg["model"]["backbone"] = args.backbone
    if not Path(args.checkpoint).exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    device = get_device()
    print(f"Using device: {device}")

    model, backbone_kwargs = build_backbone(cfg)
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    model = model.to(device)

    dataset = OpenIDataset(
        cfg["data"]["indiana_dir"],
        split=args.split,
        **backbone_kwargs["val"],
        **openi_kwargs(cfg),
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    image_embeddings, text_embeddings = encode_dataset(model, loader, device)
    similarity = (image_embeddings @ text_embeddings.T).numpy()

    i2t = recall_at_k(similarity, [1, 5, 10])
    t2i = recall_at_k(similarity.T, [1, 5, 10])
    stats = matched_random_stats(similarity)

    write_report(Path(args.out), args.checkpoint, args.split, len(dataset), i2t, t2i, stats)

    print("\nImage -> Text")
    for key, value in i2t.items():
        print(f"  {key}: {value:.2f}")
    print("\nText -> Image")
    for key, value in t2i.items():
        print(f"  {key}: {value:.2f}")
    print("\nMatched vs random")
    print(f"  matched mean: {stats['matched_mean']:.4f}")
    print(f"  random mean:  {stats['random_mean']:.4f}")
    print(f"  gap:          {stats['gap']:.4f}")
    print(f"  gap/std:      {stats['gap_normalized']:.4f}")
    print(f"\nWrote report -> {args.out}")


if __name__ == "__main__":
    main()
