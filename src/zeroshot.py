"""Zero-shot disease classification on NIH ChestX-ray14.

Works for both backbones (`--backbone medclip|biomedclip`). `--prompt all` runs
every template and prints the per-disease AUC table used for the prompt ablation.
"""
import argparse

import numpy as np
import torch
import yaml
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Subset

from dataset import NIHDataset
from prompts import build_prompts
from train import build_backbone


PROMPT_KEYS = ["simple", "findings", "clinical", "patient", "radiologist", "ensemble", "pos_neg"]


def build_prompt_tokenizer(val_kwargs: dict):
    """Return texts -> (input_ids, attention_mask) for the configured backbone."""
    tokenize_fn = val_kwargs.get("tokenize_fn")
    if tokenize_fn is not None:
        def tokenize(texts: list[str]):
            enc = tokenize_fn(list(texts))
            return enc["input_ids"], enc["attention_mask"]
        return tokenize

    tokenizer = val_kwargs["tokenizer"]

    def tokenize(texts: list[str]):
        enc = tokenizer(list(texts), padding=True, truncation=True,
                        max_length=128, return_tensors="pt")
        return enc["input_ids"], enc["attention_mask"]
    return tokenize


@torch.no_grad()
def encode_prompts(model, tokenize, diseases: list[str], prompt_key: str, device) -> torch.Tensor:
    """Returns (num_diseases, embed_dim) tensor of averaged prompt embeddings."""
    embs = []
    for disease in diseases:
        prompts = build_prompts(disease)[prompt_key]
        input_ids, attention_mask = tokenize(prompts)
        e = model.encode_text(input_ids.to(device), attention_mask.to(device))
        embs.append(e.mean(dim=0))
    return torch.stack(embs)  # (num_diseases, D)


@torch.no_grad()
def encode_images(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    """Encode once and reuse across prompt templates -- the images dominate runtime."""
    model.eval()
    embs, labels = [], []
    for batch in loader:
        embs.append(model.encode_image(batch["image"].to(device)).cpu().numpy())
        labels.append(batch["labels"].numpy())
    return np.concatenate(embs), np.concatenate(labels)


def auc_table(img_embs: np.ndarray, labels: np.ndarray, text_embs: torch.Tensor,
              diseases: list[str]) -> dict[str, float]:
    logits = img_embs @ text_embs.cpu().numpy().T
    aucs = {}
    for i, cls in enumerate(diseases):
        if labels[:, i].sum() > 0:
            aucs[cls] = float(roc_auc_score(labels[:, i], logits[:, i]))
    aucs["macro_avg"] = float(np.mean(list(aucs.values())))
    return aucs


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def write_report(out_path: str, checkpoint: str, backbone: str, n_samples: int,
                 diseases: list[str], results: dict[str, dict[str, float]]) -> None:
    header = "| Prompt | " + " | ".join(diseases) + " | Macro AUC |"
    lines = [
        "# Zero-shot classification on NIH ChestX-ray14",
        "",
        f"- Backbone: `{backbone}`",
        f"- Checkpoint: `{checkpoint}`",
        f"- Samples: `{n_samples}`",
        f"- Subset: same seed-42 selection as the published CLIP+ClinicalBERT run",
        "",
        header,
        "|" + "---|" * (len(diseases) + 2),
    ]
    for key, aucs in results.items():
        row = " | ".join(f"{aucs.get(d, float('nan')):.4f}" for d in diseases)
        lines.append(f"| {key} | {row} | {aucs['macro_avg']:.4f} |")
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",     default="configs/base.yaml")
    parser.add_argument("--checkpoint", default="checkpoints/best.pt")
    parser.add_argument("--prompt",     default="ensemble",
                        choices=PROMPT_KEYS + ["all"])
    parser.add_argument("--backbone",   choices=["medclip", "biomedclip"], default=None)
    parser.add_argument("--nih-dir",    default=None)
    parser.add_argument("--sample",     type=int, default=None,
                        help="Evaluate on a random subset of N images. With the default seed this "
                             "reproduces the exact 2000-image subset used for the published numbers.")
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--out",        default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    if args.backbone:
        cfg["model"]["backbone"] = args.backbone
    if args.nih_dir:
        cfg["data"]["nih_dir"] = args.nih_dir

    device = get_device()
    print(f"Using device: {device}", flush=True)

    model, backbone_kwargs = build_backbone(cfg)
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    model = model.to(device)
    val_kwargs = backbone_kwargs["val"]

    diseases = cfg["eval"]["nih_classes"]
    nih_kwargs = ({"transform": val_kwargs["transform"]} if val_kwargs.get("transform") is not None
                  else {"image_size": cfg["data"]["image_size"],
                        "image_normalization": cfg["data"].get("image_normalization", "imagenet")})
    dataset = NIHDataset(cfg["data"]["nih_dir"], classes=diseases, **nih_kwargs)
    if args.sample and args.sample < len(dataset):
        # Same selection as notebooks/03_evaluation.ipynb, so AUCs stay comparable
        # with the published CLIP+ClinicalBERT numbers.
        rng = np.random.RandomState(args.sample_seed)
        idx = rng.choice(len(dataset), size=args.sample, replace=False)
        dataset = Subset(dataset, idx)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    print(f"Encoding {len(dataset)} images ...", flush=True)
    img_embs, labels = encode_images(model, loader, device)

    tokenize = build_prompt_tokenizer(val_kwargs)
    prompt_keys = PROMPT_KEYS if args.prompt == "all" else [args.prompt]

    results = {}
    for key in prompt_keys:
        text_embs = encode_prompts(model, tokenize, diseases, key, device)
        results[key] = auc_table(img_embs, labels, text_embs, diseases)
        print(f"\nZero-shot evaluation  [prompt={key}]")
        for cls, auc in results[key].items():
            print(f"  {cls:<20} AUC = {auc:.4f}")

    if args.out:
        write_report(args.out, args.checkpoint, cfg["model"].get("backbone", "medclip"),
                     len(dataset), diseases, results)
        print(f"\nWrote report -> {args.out}")
