"""Visualize image/report embeddings for Stage 3 multimodal evaluation."""

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dataset import OpenIDataset  # noqa: E402
from model import MedCLIP  # noqa: E402


KEYWORD_GROUPS = [
    ("no acute", "No acute"),
    ("normal", "Normal"),
    ("pneumothorax", "Pneumothorax"),
    ("effusion", "Effusion"),
    ("cardiomegaly", "Cardiomegaly"),
    ("edema", "Edema"),
    ("opacity", "Opacity"),
    ("atelectasis", "Atelectasis"),
]


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--checkpoint", default="checkpoints/best.pt")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-samples", type=int, default=320)
    parser.add_argument("--indiana-dir", default=None,
                        help="Override cfg data.indiana_dir when data lives outside the repo.")
    parser.add_argument("--out", default="figures/stage3_embedding_tsne.png")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


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


def assign_group(caption: str) -> str:
    lower = caption.lower()
    for keyword, label in KEYWORD_GROUPS:
        if keyword not in lower:
            continue
        if keyword not in {"normal", "no acute"} and (
            f"no {keyword}" in lower
            or f"without {keyword}" in lower
            or f"negative for {keyword}" in lower
        ):
            continue
        if keyword in lower:
            return label
    return "Other"


@torch.no_grad()
def encode_dataset(model: MedCLIP, loader: DataLoader, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
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


def main() -> None:
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    if args.indiana_dir:
        cfg["data"]["indiana_dir"] = args.indiana_dir
    if not Path(args.checkpoint).exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    device = get_device()
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["text_encoder"])
    dataset = OpenIDataset(
        cfg["data"]["indiana_dir"],
        tokenizer,
        split=args.split,
        image_size=cfg["data"]["image_size"],
        image_normalization=cfg["data"].get("image_normalization", "imagenet"),
        **openi_kwargs(cfg),
    )
    if args.max_samples and args.max_samples < len(dataset):
        dataset.samples = dataset.samples[:args.max_samples]

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    model = MedCLIP(
        cfg["model"]["embed_dim"],
        cfg["model"]["freeze_image_layers"],
        cfg["training"]["temperature"],
    ).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))

    image_embeddings, text_embeddings = encode_dataset(model, loader, device)
    all_embeddings = torch.cat([image_embeddings, text_embeddings]).numpy().astype(np.float32)
    modalities = np.array(["Image"] * len(image_embeddings) + ["Report"] * len(text_embeddings))
    groups = np.array([assign_group(s["caption"]) for s in dataset.samples] * 2)

    perplexity = min(30, max(5, (len(all_embeddings) - 1) // 3))
    coords = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="random",
        learning_rate="auto",
        random_state=args.seed,
    ).fit_transform(all_embeddings)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    group_names = sorted(set(groups))
    cmap = plt.get_cmap("tab10")
    colors = {name: cmap(i % 10) for i, name in enumerate(group_names)}
    markers = {"Image": "o", "Report": "x"}

    plt.figure(figsize=(11, 8))
    for group in group_names:
        for modality in ["Image", "Report"]:
            mask = (groups == group) & (modalities == modality)
            if not mask.any():
                continue
            plt.scatter(
                coords[mask, 0],
                coords[mask, 1],
                s=24 if modality == "Image" else 36,
                c=[colors[group]],
                marker=markers[modality],
                alpha=0.72 if modality == "Image" else 0.9,
                label=f"{group} - {modality}",
            )

    plt.title(f"OpenI {args.split} image/report embeddings (t-SNE)")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8, frameon=False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()

    print(f"Encoded samples: {len(dataset)}")
    print(f"t-SNE points: {len(all_embeddings)}")
    print(f"Saved figure -> {out_path}")


if __name__ == "__main__":
    main()
