import argparse
import os
from pathlib import Path

import torch
import yaml
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from dataset import OpenIDataset
from loss import infonce_loss
from model import MedCLIP


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train(cfg: dict):
    device = get_device()
    print(f"Using device: {device}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["text_encoder"])
    model = MedCLIP(
        embed_dim=cfg["model"]["embed_dim"],
        freeze_image_layers=cfg["model"]["freeze_image_layers"],
    ).to(device)

    train_ds = OpenIDataset(cfg["data"]["indiana_dir"], tokenizer, split="train",
                            image_size=cfg["data"]["image_size"])
    val_ds   = OpenIDataset(cfg["data"]["indiana_dir"], tokenizer, split="val",
                            image_size=cfg["data"]["image_size"])

    train_loader = DataLoader(train_ds, batch_size=cfg["training"]["batch_size"],
                              shuffle=True,  num_workers=0, pin_memory=False)
    val_loader   = DataLoader(val_ds,   batch_size=cfg["training"]["batch_size"],
                              shuffle=False, num_workers=0, pin_memory=False)

    encoder_params    = list(model.image_encoder.parameters()) + list(model.text_encoder.parameters())
    projection_params = list(model.image_proj.parameters()) + list(model.text_proj.parameters()) + [model.logit_scale]
    optimizer = AdamW([
        {"params": encoder_params,    "lr": cfg["training"]["lr_encoders"]},
        {"params": projection_params, "lr": cfg["training"]["lr_projections"]},
    ])

    epochs = cfg["training"]["epochs"]
    warmup_steps = int(len(train_loader) * epochs * cfg["training"]["warmup_ratio"])
    scheduler = CosineAnnealingLR(optimizer, T_max=len(train_loader) * epochs)

    ckpt_dir = Path(cfg["paths"]["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            images = batch["image"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            logits_i, logits_t = model(images, input_ids, attention_mask)
            loss = infonce_loss(logits_i, logits_t)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        avg_train = total_loss / len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(device)
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                logits_i, logits_t = model(images, input_ids, attention_mask)
                val_loss += infonce_loss(logits_i, logits_t).item()
        avg_val = val_loss / len(val_loader)

        print(f"Epoch {epoch}/{epochs}  train_loss={avg_train:.4f}  val_loss={avg_val:.4f}", flush=True)

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(model.state_dict(), ckpt_dir / "best.pt")
            print(f"  → saved best checkpoint (val_loss={best_val_loss:.4f})", flush=True)

    torch.save(model.state_dict(), ckpt_dir / "last.pt")
    print("Training complete.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    train(cfg)
