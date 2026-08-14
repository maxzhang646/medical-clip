"""Test whether near-duplicate reports explain OpenI retrieval failures.

The Stage 3 notes guessed that boilerplate normal reports ("no acute
cardiopulmonary disease") make exact-match retrieval impossible for a subset of
the test set. That is a testable claim, not a caveat: if it holds, rank should
degrade with the number of near-duplicate reports a query has to compete with,
and failures should be dominated by cases where the top-ranked report says
essentially the same thing as the true one.

Near-duplicate similarity is computed on TF-IDF text, independent of the model,
so it cannot be an artifact of the embedding being evaluated.
"""
import argparse
from pathlib import Path
import re
import sys

import numpy as np
import torch
import yaml
from sklearn.feature_extraction.text import TfidfVectorizer
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dataset import OpenIDataset  # noqa: E402
from train import build_backbone  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--indiana-dir", default=None)
    parser.add_argument("--dup-threshold", type=float, default=0.8,
                        help="TF-IDF cosine above which two reports count as near-duplicates.")
    parser.add_argument("--examples", type=int, default=6)
    parser.add_argument("--out", default="stage6_failure_analysis.md")
    return parser.parse_args()


NORMAL_RE = re.compile(
    r"no acute\s+(cardiopulmonary\s+)?(abnormalit|disease|finding|process)"
    r"|no active (pulmonary )?disease"
    r"|normal chest"
    r"|no evidence of acute",
    re.IGNORECASE,
)


def is_normal(caption: str) -> bool:
    """Coarse semantic label. TF-IDF misses that two differently-worded normal reports
    say the same thing, so this checks agreement at the level a clinician would care
    about first: does the retrieved report call the chest normal, like the true one?"""
    return bool(NORMAL_RE.search(caption))


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def openi_kwargs(cfg: dict) -> dict:
    d = cfg["data"]
    return {
        "split_dir": d.get("split_dir"),
        "split_ratios": (d.get("train_split", 0.8), d.get("val_split", 0.1), d.get("test_split", 0.1)),
        "split_seed": d.get("split_seed", 42),
    }


@torch.no_grad()
def encode(cfg: dict, checkpoint: str, split: str, batch_size: int, device):
    model, backbone_kwargs = build_backbone(cfg)
    model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    model = model.to(device).eval()
    dataset = OpenIDataset(cfg["data"]["indiana_dir"], split=split,
                           **backbone_kwargs["val"], **openi_kwargs(cfg))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    imgs, txts = [], []
    for batch in loader:
        imgs.append(model.encode_image(batch["image"].to(device)).cpu())
        txts.append(model.encode_text(batch["input_ids"].to(device),
                                      batch["attention_mask"].to(device)).cpu())
    sim = (torch.cat(imgs) @ torch.cat(txts).T).numpy()
    return sim, [s["caption"] for s in dataset.samples], [s["uid"] for s in dataset.samples]


def main() -> None:
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    if args.indiana_dir:
        cfg["data"]["indiana_dir"] = args.indiana_dir

    device = get_device()
    print(f"Using device: {device}", flush=True)
    sim, captions, uids = encode(cfg, args.checkpoint, args.split, args.batch_size, device)
    n = len(captions)

    order = np.argsort(-sim, axis=1)
    ranks = np.argwhere(order == np.arange(n)[:, None])[:, 1] + 1
    top1 = order[:, 0]

    tfidf = TfidfVectorizer(stop_words="english", sublinear_tf=True).fit_transform(captions)
    text_sim = (tfidf @ tfidf.T).toarray()
    np.fill_diagonal(text_sim, 0.0)
    n_dups = (text_sim >= args.dup_threshold).sum(axis=1)

    lines = [f"# Do near-duplicate reports explain retrieval failure? ({args.split} split)", "",
             f"- Checkpoint: `{args.checkpoint}`",
             f"- Near-duplicate threshold: TF-IDF cosine >= {args.dup_threshold}",
             f"- Queries: {n}", ""]

    lines += ["## Rank by number of near-duplicate competitors", "",
              "| near-duplicates | queries | median rank | R@5 |", "|---|---|---|---|"]
    buckets = [(0, 0), (1, 2), (3, 5), (6, 10**6)]
    print("\nRank by number of near-duplicate competitors")
    for lo, hi in buckets:
        mask = (n_dups >= lo) & (n_dups <= hi)
        if not mask.any():
            continue
        label = f"{lo}" if lo == hi else (f"{lo}-{hi}" if hi < 10**6 else f"{lo}+")
        row = (f"| {label} | {int(mask.sum())} | {np.median(ranks[mask]):.1f} | "
               f"{(ranks[mask] <= 5).mean()*100:.1f}% |")
        lines.append(row)
        print("  " + row.strip("| ").replace(" | ", "  "))

    corr = float(np.corrcoef(n_dups, ranks)[0, 1])
    lines += ["", f"Correlation between near-duplicate count and rank: `{corr:+.3f}`", ""]
    print(f"\ncorrelation(near-duplicates, rank) = {corr:+.3f}")

    # Among failures, is the top-ranked report saying the same thing as the truth?
    failed = ranks > 5
    top1_text_sim = text_sim[np.arange(n), top1]
    near_miss = failed & (top1_text_sim >= args.dup_threshold)
    lines += ["## Are failures wrong, or just ambiguous?", "",
              "| group | queries | share |", "|---|---|---|",
              f"| retrieved the true report in top-5 | {int((~failed).sum())} | {(~failed).mean()*100:.1f}% |",
              f"| failed, but rank-1 report is a near-duplicate of the true one | {int(near_miss.sum())} | {near_miss.mean()*100:.1f}% |",
              f"| failed, rank-1 report is genuinely different | {int((failed & ~near_miss).sum())} | {(failed & ~near_miss).mean()*100:.1f}% |",
              "",
              f"Median TF-IDF similarity between the rank-1 report and the true report, over failures: "
              f"`{np.median(top1_text_sim[failed]):.3f}`", ""]
    print(f"\nfailures where rank-1 report is a near-duplicate of the truth: "
          f"{int(near_miss.sum())} / {int(failed.sum())}")
    print(f"median text similarity(rank-1, truth) over failures: {np.median(top1_text_sim[failed]):.3f}")

    normal = np.array([is_normal(c) for c in captions])
    agree = normal[np.arange(n)] == normal[top1]
    p_norm = normal.mean()
    chance = p_norm ** 2 + (1 - p_norm) ** 2
    lines += ["## Coarse semantic agreement (normal vs abnormal)", "",
              f"- Reports the regex calls normal: {int(normal.sum())} / {n} ({p_norm*100:.1f}%)",
              f"- Rank-1 shares the true report's normal/abnormal label: "
              f"{agree.mean()*100:.1f}% overall, {agree[failed].mean()*100:.1f}% among failures",
              f"- Chance agreement given the base rate: {chance*100:.1f}%",
              "",
              "TF-IDF similarity is lexical: two normal reports worded differently score low even",
              "though they say the same thing. This label is the crude semantic backstop.", ""]
    print(f"\nnormal reports: {int(normal.sum())}/{n} ({p_norm*100:.1f}%)")
    print(f"rank-1 agrees on normal/abnormal: {agree.mean()*100:.1f}% overall, "
          f"{agree[failed].mean()*100:.1f}% among failures (chance {chance*100:.1f}%)")

    lines += ["## Failure examples with the highest text similarity to the true report", ""]
    cand = np.where(failed)[0]
    for i in cand[np.argsort(-top1_text_sim[cand])][:args.examples]:
        lines += [f"### query {i} (uid {uids[i]}) — true rank {ranks[i]}, "
                  f"text similarity to rank-1 {top1_text_sim[i]:.3f}", "",
                  f"**True report:** {captions[i]}", "",
                  f"**Rank-1 retrieved:** {captions[top1[i]]}", ""]

    Path(args.out).write_text("\n".join(lines) + "\n")
    print(f"\nWrote -> {args.out}")


if __name__ == "__main__":
    main()
