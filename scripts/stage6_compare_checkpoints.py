"""Paired rank comparison between two checkpoints on the same OpenI test split.

Selecting example cases by one model's hits and misses biases the comparison: the
other model then looks worse on the first model's wins and better on its losses,
purely by regression to the mean. This script instead compares the ground-truth
rank of *every* test query under both checkpoints, and only afterwards picks
examples from strata defined by the rank change.
"""
import argparse
from pathlib import Path
import sys

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dataset import OpenIDataset  # noqa: E402
from train import build_backbone  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-a", required=True)
    parser.add_argument("--checkpoint-a", required=True)
    parser.add_argument("--label-a", default="A")
    parser.add_argument("--config-b", required=True)
    parser.add_argument("--checkpoint-b", required=True)
    parser.add_argument("--label-b", default="B")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--indiana-dir", default=None)
    parser.add_argument("--examples", type=int, default=5,
                        help="Examples to list per stratum (most improved / most degraded).")
    parser.add_argument("--out", default="stage6_checkpoint_comparison.md")
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


@torch.no_grad()
def similarity_matrix(config_path: str, checkpoint: str, split: str, batch_size: int,
                      indiana_dir, device) -> tuple[np.ndarray, list[dict]]:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    if indiana_dir:
        cfg["data"]["indiana_dir"] = indiana_dir

    model, backbone_kwargs = build_backbone(cfg)
    model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    model = model.to(device).eval()

    dataset = OpenIDataset(cfg["data"]["indiana_dir"], split=split,
                           **backbone_kwargs["val"], **openi_kwargs(cfg))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    images, texts = [], []
    for batch in loader:
        images.append(model.encode_image(batch["image"].to(device)).cpu())
        texts.append(model.encode_text(batch["input_ids"].to(device),
                                       batch["attention_mask"].to(device)).cpu())
    sim = (torch.cat(images) @ torch.cat(texts).T).numpy()
    del model
    return sim, dataset.samples


def ground_truth_ranks(sim: np.ndarray) -> np.ndarray:
    order = np.argsort(-sim, axis=1)
    gt = np.arange(sim.shape[0])
    return np.argwhere(order == gt[:, None])[:, 1] + 1


def summarize(ranks_a: np.ndarray, ranks_b: np.ndarray, label_a: str, label_b: str) -> list[str]:
    delta = ranks_b - ranks_a
    n = len(delta)
    improved = int((delta < 0).sum())
    degraded = int((delta > 0).sum())
    tied = int((delta == 0).sum())
    # Sign test: under no difference, improvements are Binomial(improved+degraded, 0.5).
    moved = improved + degraded
    se = np.sqrt(moved * 0.25)
    z = (improved - moved / 2) / se if moved else float("nan")
    rows = [
        f"| queries | {n} |",
        f"| improved under {label_b} | {improved} ({improved/n*100:.1f}%) |",
        f"| degraded under {label_b} | {degraded} ({degraded/n*100:.1f}%) |",
        f"| unchanged | {tied} |",
        f"| median rank, {label_a} | {np.median(ranks_a):.1f} |",
        f"| median rank, {label_b} | {np.median(ranks_b):.1f} |",
        f"| median rank change | {np.median(delta):+.1f} |",
        f"| sign test z | {z:+.2f} |",
    ]
    # The sign test counts every rank movement, including ones far outside the top-K
    # that no retrieval metric cares about. R@K is a threshold, so the matched test
    # for it is McNemar on the "made it into top K" indicator.
    for k in (1, 5, 10):
        in_a, in_b = ranks_a <= k, ranks_b <= k
        lost = int((in_a & ~in_b).sum())
        gained = int((~in_a & in_b).sum())
        disc = lost + gained
        chi2 = (abs(gained - lost) - 1) ** 2 / disc if disc else float("nan")
        rows.append(f"| R@{k}: {in_a.mean()*100:.2f} -> {in_b.mean()*100:.2f}, "
                    f"gained {gained} / lost {lost}, McNemar chi2 {chi2:.2f} |")
    return rows


def main() -> None:
    args = parse_args()
    device = get_device()
    print(f"Using device: {device}", flush=True)

    sim_a, samples = similarity_matrix(args.config_a, args.checkpoint_a, args.split,
                                       args.batch_size, args.indiana_dir, device)
    sim_b, _ = similarity_matrix(args.config_b, args.checkpoint_b, args.split,
                                 args.batch_size, args.indiana_dir, device)

    lines = [f"# Paired checkpoint comparison ({args.split} split)", "",
             f"- A: `{args.checkpoint_a}` ({args.label_a})",
             f"- B: `{args.checkpoint_b}` ({args.label_b})", ""]

    for direction, a, b in [("Image -> Text", sim_a, sim_b),
                            ("Text -> Image", sim_a.T, sim_b.T)]:
        ranks_a, ranks_b = ground_truth_ranks(a), ground_truth_ranks(b)
        lines += [f"## {direction}", "", "| metric | value |", "|---|---|"]
        lines += summarize(ranks_a, ranks_b, args.label_a, args.label_b)
        lines.append("")

        delta = ranks_b - ranks_a
        order = np.argsort(delta)
        lines += [f"### Largest rank improvements under {args.label_b}", "",
                  "| query | uid | rank A | rank B | change |", "|---|---|---|---|---|"]
        for i in order[:args.examples]:
            lines.append(f"| {i} | {samples[i]['uid']} | {ranks_a[i]} | {ranks_b[i]} | {delta[i]:+d} |")
        lines += ["", f"### Largest rank degradations under {args.label_b}", "",
                  "| query | uid | rank A | rank B | change |", "|---|---|---|---|---|"]
        for i in order[::-1][:args.examples]:
            lines.append(f"| {i} | {samples[i]['uid']} | {ranks_a[i]} | {ranks_b[i]} | {delta[i]:+d} |")
        lines.append("")

        print(f"\n{direction}")
        for line in summarize(ranks_a, ranks_b, args.label_a, args.label_b):
            print("  " + line.strip("| ").replace(" | ", ": "))

    Path(args.out).write_text("\n".join(lines) + "\n")
    print(f"\nWrote -> {args.out}")


if __name__ == "__main__":
    main()
