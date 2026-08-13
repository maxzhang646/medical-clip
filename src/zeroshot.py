"""Zero-shot disease classification on NIH ChestX-ray14."""
import argparse

import numpy as np
import torch
import yaml
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from dataset import NIH_CLASSES, NIHDataset
from model import MedCLIP
from prompts import build_prompts


@torch.no_grad()
def encode_prompts(model, tokenizer, diseases: list[str], prompt_key: str, device) -> torch.Tensor:
    """Returns (num_diseases, embed_dim) tensor of averaged prompt embeddings."""
    embs = []
    for disease in diseases:
        prompts = build_prompts(disease)[prompt_key]
        enc = tokenizer(prompts, padding=True, truncation=True,
                        max_length=128, return_tensors="pt").to(device)
        e = model.encode_text(enc["input_ids"], enc["attention_mask"])
        embs.append(e.mean(dim=0))
    return torch.stack(embs)  # (num_diseases, D)


@torch.no_grad()
def evaluate(model, tokenizer, loader, diseases, prompt_key, device) -> dict[str, float]:
    model.eval()
    text_embs = encode_prompts(model, tokenizer, diseases, prompt_key, device)  # (C, D)

    all_logits, all_labels = [], []
    for batch in loader:
        img_emb = model.encode_image(batch["image"].to(device))      # (B, D)
        logits  = img_emb @ text_embs.T                              # (B, C)
        all_logits.append(logits.cpu().numpy())
        all_labels.append(batch["labels"].numpy())

    logits = np.concatenate(all_logits)
    labels = np.concatenate(all_labels)

    aucs = {}
    for i, cls in enumerate(diseases):
        if labels[:, i].sum() > 0:
            aucs[cls] = roc_auc_score(labels[:, i], logits[:, i])
    aucs["macro_avg"] = np.mean(list(aucs.values()))
    return aucs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",     default="configs/base.yaml")
    parser.add_argument("--checkpoint", default="checkpoints/best.pt")
    parser.add_argument("--prompt",     default="ensemble",
                        choices=["simple", "findings", "clinical", "patient",
                                 "radiologist", "ensemble", "pos_neg"])
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("mps" if torch.backends.mps.is_available() else
                          "cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["text_encoder"])
    model = MedCLIP(
        cfg["model"]["embed_dim"],
        cfg["model"]["freeze_image_layers"],
        cfg["training"]["temperature"],
    ).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))

    diseases = cfg["eval"]["nih_classes"]
    dataset  = NIHDataset(
        cfg["data"]["nih_dir"],
        classes=diseases,
        image_size=cfg["data"]["image_size"],
        image_normalization=cfg["data"].get("image_normalization", "imagenet"),
    )
    loader   = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=4)

    print(f"\nZero-shot evaluation  [prompt={args.prompt}]")
    aucs = evaluate(model, tokenizer, loader, diseases, args.prompt, device)
    for cls, auc in aucs.items():
        print(f"  {cls:<20} AUC = {auc:.4f}")
