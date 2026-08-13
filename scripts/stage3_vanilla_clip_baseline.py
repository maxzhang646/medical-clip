"""Evaluate vanilla OpenAI CLIP on OpenI bidirectional retrieval."""

import argparse
from pathlib import Path
import sys

import clip
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dataset import OpenIDataset  # noqa: E402


class _DummyTokenizer:
    def __call__(self, text, **kwargs):
        return {
            "input_ids": torch.empty(1, 0, dtype=torch.long),
            "attention_mask": torch.empty(1, 0, dtype=torch.long),
        }


class CLIPOpenIDataset(Dataset):
    def __init__(self, samples: list[dict], preprocess):
        self.samples = samples
        self.preprocess = preprocess

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        image = Image.open(sample["image_path"]).convert("RGB")
        return {
            "image": self.preprocess(image),
            "caption": sample["caption"],
            "uid": sample["uid"],
            "image_path": sample["image_path"],
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--model-name", default="ViT-B/32")
    parser.add_argument("--indiana-dir", default=None)
    parser.add_argument("--out", default="stage3_vanilla_clip_baseline.md")
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


def collate_clip(batch: list[dict]) -> dict:
    return {
        "image": torch.stack([item["image"] for item in batch]),
        "caption": [item["caption"] for item in batch],
        "uid": [item["uid"] for item in batch],
        "image_path": [item["image_path"] for item in batch],
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
    return {
        "matched_mean": float(matched.mean()),
        "matched_std": float(matched.std()),
        "random_mean": float(random_scores.mean()),
        "random_std": float(random_scores.std()),
        "gap": float(matched.mean() - random_scores.mean()),
    }


@torch.no_grad()
def encode_dataset(model, loader: DataLoader, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    image_embeddings = []
    text_embeddings = []
    for batch in loader:
        images = batch["image"].to(device)
        text_tokens = clip.tokenize(batch["caption"], truncate=True).to(device)

        image_features = model.encode_image(images).float()
        text_features = model.encode_text(text_tokens).float()
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        image_embeddings.append(image_features.cpu())
        text_embeddings.append(text_features.cpu())
    return torch.cat(image_embeddings), torch.cat(text_embeddings)


def write_report(
    out_path: Path,
    model_name: str,
    split: str,
    sample_count: int,
    i2t: dict[str, float],
    t2i: dict[str, float],
    stats: dict[str, float],
) -> None:
    lines = [
        "# Stage 3 Vanilla OpenAI CLIP Baseline",
        "",
        "This run evaluates the original OpenAI CLIP dual tower without medical fine-tuning.",
        "",
        f"- Model: `{model_name}`",
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
        "",
        "Interpretation: this is the aligned general-domain CLIP baseline. It is useful because both towers were pretrained together, even though the pretraining data is not medical-domain specific.",
    ]
    out_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    if args.indiana_dir:
        cfg["data"]["indiana_dir"] = args.indiana_dir

    device = get_device()
    print(f"Using device: {device}")
    model, preprocess = clip.load(args.model_name, device=device)
    model.eval()

    split_dataset = OpenIDataset(
        cfg["data"]["indiana_dir"],
        _DummyTokenizer(),
        split=args.split,
        image_size=cfg["data"]["image_size"],
        **openi_kwargs(cfg),
    )
    dataset = CLIPOpenIDataset(split_dataset.samples, preprocess)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_clip,
    )

    image_embeddings, text_embeddings = encode_dataset(model, loader, device)
    similarity = (image_embeddings @ text_embeddings.T).numpy()

    i2t = recall_at_k(similarity, [1, 5, 10])
    t2i = recall_at_k(similarity.T, [1, 5, 10])
    stats = matched_random_stats(similarity)

    write_report(Path(args.out), args.model_name, args.split, len(dataset), i2t, t2i, stats)

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
    print(f"\nWrote report -> {args.out}")


if __name__ == "__main__":
    main()
