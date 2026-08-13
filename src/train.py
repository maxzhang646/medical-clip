import argparse
import os
from pathlib import Path

import torch
import yaml
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from dataset import OpenIDataset
from loss import infonce_loss
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


def build_backbone(cfg: dict):
    """Return (model, per-split dataset kwargs) for the configured backbone.

    Defaults to the original CLIP + ClinicalBERT MedCLIP, so existing configs
    behave exactly as before.
    """
    model_cfg = cfg["model"]
    backbone = model_cfg.get("backbone", "medclip")

    if backbone == "medclip":
        tokenizer = AutoTokenizer.from_pretrained(model_cfg["text_encoder"])
        model = MedCLIP(
            embed_dim=model_cfg["embed_dim"],
            freeze_image_layers=model_cfg["freeze_image_layers"],
            temperature=cfg["training"]["temperature"],
        )
        shared = {
            "tokenizer": tokenizer,
            "image_size": cfg["data"]["image_size"],
            "image_normalization": cfg["data"].get("image_normalization", "imagenet"),
        }
        return model, {"train": shared, "val": shared}

    if backbone == "biomedclip":
        from biomedclip_model import (
            DEFAULT_CONTEXT_LENGTH, DEFAULT_MODEL, BioMedCLIPFinetune, build_tokenize_fn,
        )
        model_name = model_cfg.get("model_name", DEFAULT_MODEL)
        context_length = model_cfg.get("context_length", DEFAULT_CONTEXT_LENGTH)
        model = BioMedCLIPFinetune(
            model_name=model_name,
            freeze_image_layers=model_cfg["freeze_image_layers"],
            freeze_text_layers=model_cfg.get("freeze_text_layers", 0),
        )
        print(f"BioMedCLIP: {model_name}", flush=True)
        print(f"  frozen image blocks: {model.frozen_image_layers}, "
              f"frozen text layers: {model.frozen_text_layers}", flush=True)
        print("  using BioMedCLIP preprocessing and tokenizer "
              "(data.image_size / image_normalization are ignored)", flush=True)
        tokenize_fn = build_tokenize_fn(model_name, context_length)
        return model, {
            "train": {"tokenizer": None, "transform": model.preprocess_train,
                      "tokenize_fn": tokenize_fn},
            "val":   {"tokenizer": None, "transform": model.preprocess_val,
                      "tokenize_fn": tokenize_fn},
        }

    raise ValueError(f"Unknown model.backbone: {backbone}. Valid options: medclip, biomedclip")


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train(cfg: dict):
    device = get_device()
    print(f"Using device: {device}", flush=True)

    model, backbone_kwargs = build_backbone(cfg)
    model = model.to(device)

    dataset_kwargs = openi_kwargs(cfg)
    train_ds = OpenIDataset(cfg["data"]["indiana_dir"], split="train",
                            **backbone_kwargs["train"], **dataset_kwargs)
    val_ds   = OpenIDataset(cfg["data"]["indiana_dir"], split="val",
                            **backbone_kwargs["val"], **dataset_kwargs)

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
    total_steps = len(train_loader) * epochs
    warmup_steps = int(total_steps * cfg["training"]["warmup_ratio"])
    if warmup_steps > 0:
        warmup = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_steps)
        cosine = CosineAnnealingLR(optimizer, T_max=max(1, total_steps - warmup_steps))
        scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps])
    else:
        scheduler = CosineAnnealingLR(optimizer, T_max=total_steps)

    # Mixed precision: only meaningful on CUDA, so MPS/CPU runs silently stay fp32.
    use_amp = bool(cfg["training"].get("fp16", False)) and device.type == "cuda"
    if hasattr(torch.amp, "GradScaler"):
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
        autocast = lambda: torch.amp.autocast("cuda", enabled=use_amp)  # noqa: E731
    else:  # torch < 2.4
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
        autocast = lambda: torch.cuda.amp.autocast(enabled=use_amp)  # noqa: E731
    print(f"Mixed precision: {'on' if use_amp else 'off'}", flush=True)

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

            with autocast():
                logits_i, logits_t = model(images, input_ids, attention_mask)
                loss = infonce_loss(logits_i, logits_t)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)  # clip on real gradients, not scaled ones
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
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
                with autocast():
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
    parser.add_argument("--indiana-dir", default=None,
                        help="Override cfg data.indiana_dir when data lives outside the repo.")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    if args.indiana_dir:
        cfg["data"]["indiana_dir"] = args.indiana_dir

    train(cfg)
