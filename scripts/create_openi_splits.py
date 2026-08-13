"""Create deterministic OpenI train/val/test UID split files."""

import argparse
from pathlib import Path

import pandas as pd
import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--indiana-dir", default=None,
                        help="Override cfg data.indiana_dir when data lives outside the repo.")
    parser.add_argument("--out-dir", default=None,
                        help="Override cfg data.split_dir.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    data_cfg = cfg["data"]
    root = Path(args.indiana_dir or data_cfg["indiana_dir"])
    out_dir = Path(args.out_dir or data_cfg.get("split_dir", "splits"))
    out_dir.mkdir(parents=True, exist_ok=True)

    reports = pd.read_csv(root / "indiana_reports.csv")
    projections = pd.read_csv(root / "indiana_projections.csv")
    frontal = projections[projections["projection"] == "Frontal"][["uid", "filename"]]
    df = reports.merge(frontal, on="uid", how="inner")
    df = df[df["findings"].notna() & df["impression"].notna()]
    df = df[df["findings"].str.strip().ne("") & df["impression"].str.strip().ne("")]
    df = df.drop_duplicates(subset="uid").reset_index(drop=True)

    uids = df["uid"].unique()
    rng = pd.Series(uids).sample(frac=1, random_state=data_cfg.get("split_seed", 42)).values
    train_ratio = data_cfg.get("train_split", 0.8)
    val_ratio = data_cfg.get("val_split", 0.1)
    test_ratio = data_cfg.get("test_split", 0.1)
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"Split ratios must sum to 1.0, got {total}")

    train_cut = int(len(rng) * train_ratio)
    val_cut = int(len(rng) * (train_ratio + val_ratio))
    splits = {
        "train": rng[:train_cut],
        "val": rng[train_cut:val_cut],
        "test": rng[val_cut:],
    }

    for name, split_uids in splits.items():
        path = out_dir / f"openi_{name}_uids.txt"
        path.write_text("\n".join(str(int(uid)) for uid in split_uids) + "\n")
        print(f"Wrote {len(split_uids):4d} UIDs -> {path}")


if __name__ == "__main__":
    main()
