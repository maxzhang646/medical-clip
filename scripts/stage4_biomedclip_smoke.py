"""Smoke check for the BioMedCLIP fine-tuning backbone.

Verifies, on a small real OpenI batch:
  1. BioMedCLIP loads and its own preprocessing/tokenizer flow through OpenIDataset
  2. forward + InfoNCE + backward runs, and frozen blocks receive no gradient
  3. the default MedCLIP dataset path is unchanged (regression guard)

Usage:
  python3 scripts/stage4_biomedclip_smoke.py --indiana-dir /path/to/openi
"""
import argparse
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from biomedclip_model import BioMedCLIPFinetune, build_tokenize_fn  # noqa: E402
from dataset import OpenIDataset  # noqa: E402
from loss import infonce_loss  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--indiana-dir", required=True)
    parser.add_argument("--split-dir", default="splits")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--freeze-image-layers", type=int, default=8)
    parser.add_argument("--skip-medclip-check", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("[1/3] Loading BioMedCLIP ...", flush=True)
    model = BioMedCLIPFinetune(freeze_image_layers=args.freeze_image_layers)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  frozen image blocks : {model.frozen_image_layers}")
    print(f"  trainable params    : {trainable/1e6:.1f}M / {total/1e6:.1f}M")
    print(f"  logit_scale         : {model.logit_scale.exp().item():.2f} "
          f"(temperature {1/model.logit_scale.exp().item():.4f})")

    print("[2/3] Building OpenI train split with BioMedCLIP preprocessing ...", flush=True)
    ds = OpenIDataset(
        args.indiana_dir,
        tokenizer=None,
        split="train",
        split_dir=args.split_dir,
        transform=model.preprocess_train,
        tokenize_fn=build_tokenize_fn(),
    )
    print(f"  samples             : {len(ds)}")
    sample = ds[0]
    print(f"  image tensor        : {tuple(sample['image'].shape)} "
          f"[{sample['image'].min():.3f}, {sample['image'].max():.3f}]")
    print(f"  input_ids           : {tuple(sample['input_ids'].shape)} "
          f"(non-pad {int(sample['attention_mask'].sum())})")

    batch = [ds[i] for i in range(args.batch_size)]
    images = torch.stack([b["image"] for b in batch])
    input_ids = torch.stack([b["input_ids"] for b in batch])
    attention_mask = torch.stack([b["attention_mask"] for b in batch])

    logits_i, logits_t = model(images, input_ids, attention_mask)
    loss = infonce_loss(logits_i, logits_t)
    loss.backward()

    expected = torch.log(torch.tensor(float(args.batch_size))).item()
    print(f"  logits              : {tuple(logits_i.shape)}")
    print(f"  InfoNCE loss        : {loss.item():.4f} (random baseline {expected:.4f})")

    blocks = list(model.model.visual.trunk.blocks)
    frozen_grads = [p.grad for b in blocks[:model.frozen_image_layers] for p in b.parameters()]
    live_grads = [p.grad for b in blocks[model.frozen_image_layers:] for p in b.parameters()]
    assert all(g is None for g in frozen_grads), "frozen blocks received gradients"
    assert any(g is not None and g.abs().sum() > 0 for g in live_grads), "no gradient reached the unfrozen blocks"
    print(f"  grad check          : ok ({model.frozen_image_layers} frozen, "
          f"{len(blocks) - model.frozen_image_layers} training)")

    if args.skip_medclip_check:
        print("[3/3] Skipped MedCLIP regression check.")
        return

    print("[3/3] Regression check: default MedCLIP dataset path ...", flush=True)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("medicalai/ClinicalBERT")
    ref = OpenIDataset(args.indiana_dir, tok, split="train", split_dir=args.split_dir)
    ref_sample = ref[0]
    assert len(ref) == len(ds), "sample count changed between backbones"
    assert tuple(ref_sample["image"].shape) == (3, 224, 224)
    assert tuple(ref_sample["input_ids"].shape) == (128,)
    print(f"  samples             : {len(ref)} (matches)")
    print(f"  image / input_ids   : {tuple(ref_sample['image'].shape)} / "
          f"{tuple(ref_sample['input_ids'].shape)}  unchanged")
    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
