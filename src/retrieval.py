"""Bidirectional image-text retrieval evaluation (Recall@K, MedR)."""
import argparse

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from dataset import OpenIDataset
from model import MedCLIP


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


def recall_at_k(sim_matrix: np.ndarray, k_list: list[int]) -> dict[str, float]:
    n = sim_matrix.shape[0]
    ranks = np.argsort(-sim_matrix, axis=1)
    ground_truth = np.arange(n)
    results = {}
    for k in k_list:
        hits = np.any(ranks[:, :k] == ground_truth[:, None], axis=1)
        results[f"R@{k}"] = hits.mean() * 100
    results["MedR"] = float(np.median(
        np.argwhere(ranks == ground_truth[:, None])[:, 1] + 1
    ))
    return results


@torch.no_grad()
def evaluate(model, loader, device) -> dict[str, float]:
    model.eval()
    img_embs, txt_embs = [], []
    for batch in loader:
        img_embs.append(model.encode_image(batch["image"].to(device)).cpu())
        txt_embs.append(model.encode_text(
            batch["input_ids"].to(device),
            batch["attention_mask"].to(device),
        ).cpu())

    I = torch.cat(img_embs).numpy()
    T = torch.cat(txt_embs).numpy()
    sim = I @ T.T

    i2t = recall_at_k(sim,   [1, 5, 10])
    t2i = recall_at_k(sim.T, [1, 5, 10])
    return {"I→T": i2t, "T→I": t2i}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",     default="configs/base.yaml")
    parser.add_argument("--checkpoint", default="checkpoints/best.pt")
    parser.add_argument("--indiana-dir", default=None,
                        help="Override cfg data.indiana_dir when data lives outside the repo.")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    if args.indiana_dir:
        cfg["data"]["indiana_dir"] = args.indiana_dir

    device = torch.device("mps" if torch.backends.mps.is_available() else
                          "cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["text_encoder"])
    model = MedCLIP(
        cfg["model"]["embed_dim"],
        cfg["model"]["freeze_image_layers"],
        cfg["training"]["temperature"],
    ).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))

    test_ds = OpenIDataset(
        cfg["data"]["indiana_dir"],
        tokenizer,
        split="test",
        image_size=cfg["data"]["image_size"],
        image_normalization=cfg["data"].get("image_normalization", "imagenet"),
        **openi_kwargs(cfg),
    )
    loader  = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=4)

    results = evaluate(model, loader, device)
    for direction, metrics in results.items():
        print(f"\n{direction}")
        for k, v in metrics.items():
            print(f"  {k}: {v:.2f}")
