"""Generate qualitative retrieval examples for Stage 3."""

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
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--num-queries", type=int, default=8)
    parser.add_argument("--direction", default="image-to-text",
                        choices=["image-to-text", "text-to-image"])
    parser.add_argument("--selection", default="mixed", choices=["mixed", "even"],
                        help="mixed selects successful and failed retrieval cases; even samples evenly by index.")
    parser.add_argument("--query-indices", default=None,
                        help="Comma-separated query indices, overriding --selection. Required to "
                             "compare two checkpoints case by case: otherwise each model picks its "
                             "own successes and failures and the outputs are not comparable.")
    parser.add_argument("--indiana-dir", default=None,
                        help="Override cfg data.indiana_dir when data lives outside the repo.")
    parser.add_argument("--backbone", choices=["medclip", "biomedclip"], default=None,
                        help="Override cfg model.backbone.")
    parser.add_argument("--out", default="stage3_retrieval_examples.md")
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


def trim(text: str, max_chars: int = 650) -> str:
    text = " ".join(text.split())
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "..."


def write_examples(
    out_path: Path,
    dataset: OpenIDataset,
    similarity: torch.Tensor,
    query_indices: list[int],
    gt_ranks: torch.Tensor,
    recall_at_k: float,
    top_k: int,
    checkpoint: str,
    direction: str,
) -> None:
    query_name = "image" if direction == "image-to-text" else "report"
    target_name = "report" if direction == "image-to-text" else "image"
    lines = [
        "# Stage 3 Retrieval Examples",
        "",
        f"Qualitative {query_name}-to-{target_name} retrieval examples from the OpenI split.",
        "",
        f"- Checkpoint: `{checkpoint}`",
        f"- Split: `{getattr(dataset, 'split_name', 'unknown')}`",
        f"- Direction: `{direction}`",
        f"- Top-k: `{top_k}`",
        f"- Recall@{top_k}: `{recall_at_k:.2f}%`",
        "",
    ]

    for case_id, query_idx in enumerate(query_indices, start=1):
        scores = similarity[query_idx]
        top = torch.topk(scores, k=min(top_k, len(scores))).indices.tolist()
        query = dataset.samples[query_idx]
        full_rank = int(gt_ranks[query_idx].item())
        rank = full_rank if full_rank <= top_k else None

        query_block = [
            f"## Case {case_id}: Query UID {query['uid']}",
            "",
            f"- Query index: `{query_idx}`",
            f"- Ground-truth full rank: `{full_rank}`",
            f"- Ground-truth rank in top-{top_k}: `{rank if rank is not None else 'not retrieved'}`",
        ]
        if direction == "image-to-text":
            query_block.append(f"- Query image path: `{query['image_path']}`")
        query_block.extend([
            "",
            "**Query report**" if direction == "text-to-image" else "**Ground-truth report**",
            "",
            trim(query["caption"]),
            "",
            "**Retrieved images**" if direction == "text-to-image" else "**Retrieved reports**",
            "",
        ])
        lines.extend(query_block)

        for rank_idx, retrieved_idx in enumerate(top, start=1):
            retrieved = dataset.samples[retrieved_idx]
            marker = "MATCH" if retrieved_idx == query_idx else "MISMATCH"
            lines.extend([
                f"### Rank {rank_idx}: UID {retrieved['uid']} ({marker})",
                "",
                f"- Similarity: `{scores[retrieved_idx].item():.4f}`",
                f"- Dataset index: `{retrieved_idx}`",
                f"- Image path: `{retrieved['image_path']}`",
                "",
                trim(retrieved["caption"]),
                "",
            ])

    out_path.write_text("\n".join(lines) + "\n")


def compute_ground_truth_ranks(similarity: torch.Tensor) -> torch.Tensor:
    # Rank in numpy, matching scripts/stage3_medclip_diagnostic.py. torch.argsort and
    # np.argsort(-x) break float ties differently, which was enough to move one
    # borderline case out of 320 and make the two scripts disagree by 0.31pp.
    ranks = torch.from_numpy(np.argsort(-similarity.numpy(), axis=1))
    ground_truth = torch.arange(similarity.size(0))
    return (ranks == ground_truth[:, None]).nonzero()[:, 1] + 1


def select_query_indices(gt_ranks: torch.Tensor, top_k: int, num_queries: int, selection: str) -> list[int]:
    if selection == "even":
        if num_queries == 1:
            return [0]
        return torch.linspace(0, len(gt_ranks) - 1, steps=num_queries).round().long().tolist()

    hits = torch.nonzero(gt_ranks <= top_k).flatten().tolist()
    misses = torch.nonzero(gt_ranks > top_k).flatten().tolist()
    hit_count = min(len(hits), max(1, num_queries // 2))
    miss_count = min(len(misses), num_queries - hit_count)

    selected = []
    if hit_count:
        selected.extend(torch.linspace(0, len(hits) - 1, steps=hit_count).round().long().tolist())
        selected = [hits[i] for i in selected]
    if miss_count:
        miss_positions = torch.linspace(0, len(misses) - 1, steps=miss_count).round().long().tolist()
        selected.extend(misses[i] for i in miss_positions)

    return selected[:num_queries]


def main() -> None:
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    if args.indiana_dir:
        cfg["data"]["indiana_dir"] = args.indiana_dir
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
    dataset.split_name = args.split
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    image_embeddings, text_embeddings = encode_dataset(model, loader, device)
    base_similarity = image_embeddings @ text_embeddings.T
    similarity = base_similarity if args.direction == "image-to-text" else base_similarity.T
    gt_ranks = compute_ground_truth_ranks(similarity)
    recall_at_k = (gt_ranks <= args.top_k).float().mean().item() * 100

    if args.query_indices:
        query_indices = [int(i) for i in args.query_indices.split(",") if i.strip() != ""]
        out_of_range = [i for i in query_indices if not 0 <= i < len(dataset)]
        if out_of_range:
            raise ValueError(f"--query-indices out of range for {len(dataset)} samples: {out_of_range}")
    else:
        num_queries = min(args.num_queries, len(dataset))
        query_indices = select_query_indices(gt_ranks, args.top_k, num_queries, args.selection)
    print(f"Query indices: {','.join(str(i) for i in query_indices)}")

    write_examples(
        Path(args.out),
        dataset,
        similarity,
        query_indices,
        gt_ranks,
        recall_at_k,
        args.top_k,
        args.checkpoint,
        args.direction,
    )
    print(f"Recall@{args.top_k}: {recall_at_k:.2f}%")
    print(f"Wrote {len(query_indices)} retrieval examples -> {args.out}")


if __name__ == "__main__":
    main()
