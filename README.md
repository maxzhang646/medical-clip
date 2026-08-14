# Medical CLIP: Chest X-ray Vision-Language Alignment and Evidence Retrieval

A medical vision-language learning project focused on aligning chest X-ray images with radiology reports via contrastive learning. The main goal is to study multimodal representation learning through image-text retrieval, prompt-based zero-shot evaluation, and evidence retrieval for downstream medical reasoning.

**[Full experimental report →](REPORT.md)** — including four claims this project made and later
overturned with its own experiments, and the methodological traps behind them.

---

## Results

### Retrieval (OpenI test split, 320 samples)

| Model | I→T R@1 | R@5 | R@10 | MedR | T→I R@1 | R@5 | R@10 | MedR |
|-------|---------|-----|------|------|---------|-----|------|------|
| Vanilla OpenAI CLIP (zero-shot) | 0.00% | 1.56% | 3.12% | 162.5 | 0.31% | 2.81% | 4.06% | 166.0 |
| BioMedCLIP (zero-shot) | 1.56% | 5.31% | 8.12% | 120.0 | 1.25% | 6.25% | 8.75% | 108.5 |
| CLIP+ClinicalBERT (fine-tuned) | 3.75% | 12.81% | 20.62% | 49.5 | 4.38% | 11.88% | 19.38% | 47.0 |
| **BioMedCLIP (fine-tuned)** | **6.25%** | **17.50%** | **27.19%** | **45.5** | **5.31%** | **17.50%** | **23.75%** | **46.0** |

Two findings, in order of how much they move the numbers:

1. **Fine-tuning on radiology report pairs is what creates medical alignment at all.** Vanilla CLIP sits exactly at the random baseline (R@5 1.56% for 320 candidates); fine-tuning takes the same architecture to 12.81% and cuts median rank from 162 → 49.5.
2. **The starting point then matters more than the recipe** — but the retrieval table is too small to prove it. Fine-tuning BioMedCLIP on the identical 2,554 pairs with identical hyperparameters raises R@1 by 67% and R@5 by 37%. A paired McNemar test over all 320 queries puts that at p = 0.20 (42 queries gained a top-5 hit, 30 lost one): consistent in direction, not significant. The claim rests instead on cross-dataset zero-shot, where all seven prompt templates improve by 0.075–0.196 Macro AUC. An earlier version of this README made the reverse comparison — fine-tuned against zero-shot — which was unfair to BioMedCLIP. See [REPORT.md](REPORT.md) §3.1 and §4.1 for the 2x2, the paired tests, and the remaining confound (ViT-B/16 vs B/32).

### Context vs published methods

| Model | Training pairs | Zero-shot Macro AUC (NIH) |
|-------|---------------|--------------------------|
| Vanilla CLIP | 400M (general) | ~0.50 |
| Ours — CLIP+ClinicalBERT (patient prompt) | 2,554 | 0.589 |
| **Ours — BioMedCLIP FT (clinical prompt)** | **2,554** | **0.682** |
| MedCLIP — Wang et al., 2022 | ~200K | ~0.730 |
| CheXzero — Tiu et al., 2022 | ~227K | ~0.875 |

With **89× fewer training pairs** than CheXzero we reach 78% of its AUC, and we are within 0.05 of MedCLIP, which used ~78× more pairs.

### Zero-shot Classification (NIH ChestX-ray14, 2000 samples)

The prompt ablation was run on both backbones, on the same seed-42 image subset. Running it twice is
what makes it informative: a pattern that appears on one checkpoint is a property of that checkpoint
until a second one agrees.

| Prompt | CLIP+ClinicalBERT | BioMedCLIP FT | Δ |
|--------|-------------------|---------------|---|
| simple | 0.4487 | 0.6153 | +0.167 |
| findings | 0.4762 | 0.6719 | +0.196 |
| **clinical** | 0.5273 | **0.6824** | +0.155 |
| patient | **0.5889** | 0.6642 | +0.075 |
| radiologist | 0.4854 | 0.6542 | +0.169 |
| ensemble | 0.4882 | 0.6701 | +0.182 |
| pos_neg | 0.4496 | 0.5913 | +0.142 |

| Claim from the first ablation | Holds on the second backbone? |
|-------------------------------|-------------------------------|
| `patient` is the best template | **No** — `clinical` wins; `patient` falls to 4th |
| `ensemble` loses to the best single prompt | **Yes, but barely** — 3rd of 7 on both, yet the deficit shrinks from −0.101 to −0.012 |
| `pos_neg` is the worst strategy | **Yes** — last on both |

Which template wins is checkpoint-specific; the two negative results hold on both. But the ensemble
penalty nearly vanishes on the stronger backbone, so "negative interference between prompt styles" is
not a property of ensembling — it is a symptom of a weak visual representation. The same pattern shows
up as shrinking prompt sensitivity overall (spread across templates 0.140 → 0.091): prompt
engineering matters most when the visual encoder is weak.

Per-backbone detail below; full numbers, error bars, and the claims that did not survive are in [REPORT.md](REPORT.md).

#### CLIP+ClinicalBERT — AUC-ROC by prompt template

| Prompt | Atelectasis | Cardiomegaly | Consolidation | Edema | Effusion | Infiltration | Pneumonia | Pneumothorax | **Macro AUC** |
|--------|-------------|--------------|---------------|-------|----------|--------------|-----------|--------------|---------------|
| simple | 0.690 | 0.737 | 0.367 | 0.238 | 0.326 | 0.399 | 0.466 | 0.366 | 0.449 |
| findings | 0.703 | 0.730 | 0.477 | 0.264 | 0.353 | 0.370 | 0.504 | 0.409 | 0.476 |
| clinical | 0.720 | 0.710 | 0.557 | 0.455 | 0.356 | 0.488 | 0.574 | 0.358 | 0.527 |
| **patient** | **0.707** | 0.710 | **0.630** | **0.473** | **0.479** | **0.530** | **0.663** | **0.521** | **0.589** |
| radiologist | 0.712 | 0.709 | 0.536 | 0.290 | 0.369 | 0.394 | 0.521 | 0.352 | 0.485 |
| ensemble | 0.712 | **0.721** | 0.511 | 0.309 | 0.348 | 0.411 | 0.537 | 0.356 | 0.488 |

![Prompt Ablation](figures/prompt_ablation.png)

**Findings that survived the second backbone:**
- Cardiomegaly is prompt-insensitive on both models (range < 0.03 and < 0.02)
- `pos_neg` is the worst strategy on both

**Findings that did not:**
- `patient` being best is specific to this checkpoint
- The *size* of the ensemble penalty: −0.101 here, −0.012 on BioMedCLIP
- Edema was the hardest class here (best 0.473, below random) and was attributed to under-representation in OpenI. Fine-tuned BioMedCLIP reaches **0.788** on the same data, so the constraint was the visual representation, not the training reports. Infiltration is the class that is broken on both (0.374–0.530)

![ROC Curves](figures/roc_curves.png)

#### BioMedCLIP fine-tuned — AUC-ROC by prompt template

| Prompt | Atelectasis | Cardiomegaly | Consolidation | Edema | Effusion | Infiltration | Pneumonia | Pneumothorax | **Macro AUC** |
|--------|-------------|--------------|---------------|-------|----------|--------------|-----------|--------------|---------------|
| simple | 0.573 | 0.779 | 0.683 | 0.468 | 0.778 | 0.374 | 0.674 | 0.594 | 0.615 |
| findings | 0.668 | 0.777 | 0.691 | 0.754 | 0.747 | 0.479 | 0.664 | 0.596 | 0.672 |
| **clinical** | 0.636 | 0.771 | 0.642 | **0.788** | 0.770 | 0.463 | **0.708** | **0.681** | **0.682** |
| patient | **0.693** | 0.787 | 0.653 | 0.625 | 0.764 | 0.415 | 0.689 | 0.688 | 0.664 |
| radiologist | 0.691 | **0.788** | 0.668 | 0.603 | **0.782** | 0.400 | 0.643 | 0.660 | 0.654 |
| ensemble | 0.672 | 0.782 | **0.690** | 0.686 | 0.774 | 0.404 | 0.703 | 0.649 | 0.670 |
| pos_neg | 0.641 | 0.772 | 0.519 | 0.466 | 0.725 | 0.378 | 0.634 | 0.596 | 0.591 |

Per-class AUCs for rare findings carry wide error bars at 2,000 images (≈ ±0.06 for Edema, ±0.09 for
Pneumonia), so single-class differences under ~0.1 are not meaningful. Infiltration (~340 positives,
SE ≈ 0.027) is genuinely below random on every template.

---

## Method

We fine-tune a CLIP-style model on (X-ray image, radiology report) pairs using InfoNCE contrastive loss. The image encoder starts from OpenAI's pretrained ViT-B/32; the text encoder uses ClinicalBERT, pretrained on clinical notes.

```
X-ray image      →  ViT-B/32  →  Linear(512→512)  ─┐
                                                      ├─ cosine sim → InfoNCE loss (τ=0.07)
Radiology report →  ClinicalBERT  →  Linear(768→512) ─┘
```

At inference, zero-shot classification works by comparing image embeddings against text prompt embeddings — no task-specific labels needed.

**Design choices:**
- Freeze the first 8 ViT transformer blocks; fine-tune the rest + projection heads
- Use `Findings + Impression` sections of radiology reports as the paired text caption
- Patient-level train/val/test split to prevent data leakage

**Training:** Best checkpoint at Epoch 9 (val_loss = 3.640); overfitting begins thereafter, reflecting the small dataset size (2,554 training pairs).

---

## Datasets

| Dataset | Role | Size | Source |
|---------|------|------|--------|
| Indiana University OpenI | Training (image + report pairs) | 7,470 images, 3,955 reports | [Kaggle](https://www.kaggle.com/datasets/raddar/chest-xrays-indiana-university) |
| NIH ChestX-ray14 | Zero-shot evaluation only | 112,120 images, 14 disease labels | [Kaggle](https://www.kaggle.com/datasets/nih-chest-xrays/data) |

### Download

```bash
# Requires Kaggle API credentials (~/.kaggle/kaggle.json)
kaggle datasets download raddar/chest-xrays-indiana-university -p data/indiana --unzip
kaggle datasets download nih-chest-xrays/data -p data/nih --unzip
```

Expected OpenI layout:

```text
data/indiana/
├── indiana_reports.csv
├── indiana_projections.csv
└── images/
    └── images_normalized/
        └── *.png
```

Every command below assumes this layout. If OpenI lives elsewhere, either edit `data.indiana_dir` in the config or pass `--indiana-dir` to any script.

**Zero-shot disease classes (NIH):** Atelectasis · Cardiomegaly · Consolidation · Edema · Effusion · Infiltration · Pneumonia · Pneumothorax

---

## Project Structure

```
xray/
├── data/
│   ├── indiana/          # OpenI images + XML reports
│   └── nih/              # NIH images + Data_Entry_2017.csv
├── src/
│   ├── dataset.py        # OpenIDataset, NIHDataset
│   ├── model.py          # MedCLIP model class
│   ├── loss.py           # InfoNCE loss
│   ├── train.py          # training loop
│   ├── retrieval.py      # Recall@K, MedR evaluation
│   ├── zeroshot.py       # zero-shot classifier
│   └── prompts.py        # prompt template library
├── notebooks/
│   └── 03_evaluation.ipynb   # retrieval + zero-shot evaluation
├── configs/
│   └── base.yaml         # hyperparameters
├── splits/
│   ├── openi_train_uids.txt
│   ├── openi_val_uids.txt
│   └── openi_test_uids.txt
├── checkpoints/
│   └── best.pt           # best model (Epoch 9)
├── REPORT.md            # consolidated experimental report
├── docs/archive/        # per-stage documents and raw run outputs
└── requirements.txt
```

---

## Setup

```bash
git clone https://github.com/maxzhang646/medical-clip.git
cd medical-clip
pip install -r requirements.txt
```

Requirements: Python 3.10+, PyTorch 2.0+ (MPS supported for Apple Silicon).

### Reproducibility Smoke Check

After downloading OpenI into `data/indiana`, run:

```bash
python3 scripts/stage2_smoke_check.py
python3 scripts/stage1_toy_infonce.py
python3 scripts/stage1_inspect_openi_batch.py --batch-size 2 --split train
python3 scripts/stage1_forward_real_batch.py --batch-size 2 --split train
```

If OpenI lives somewhere else, add `--indiana-dir /path/to/openi` to any of these.

To regenerate deterministic OpenI split files:

```bash
python3 scripts/create_openi_splits.py
```

The repository includes UID split files under `splits/`; they make the Stage 1 examples and retrieval evaluation use stable train/val/test partitions.

### Checkpoints

`checkpoints/` is ignored by git because model weights are large. To run commands that require `checkpoints/best.pt`, either train the model or place a compatible checkpoint at that path:

```bash
python3 src/train.py --config configs/base.yaml
python3 scripts/stage1_forward_real_batch.py --checkpoint checkpoints/best.pt
```

---

## Training

```bash
python src/train.py --config configs/base.yaml
```

The training script auto-detects MPS (Apple Silicon) / CUDA / CPU.

If OpenI is outside `data/indiana`, pass:

```bash
python src/train.py --config configs/base.yaml
```

---

## Evaluation

```bash
# Retrieval on OpenI test split
python src/retrieval.py --checkpoint checkpoints/best.pt
# If OpenI is outside data/indiana:
python src/retrieval.py --checkpoint checkpoints/best.pt

# Zero-shot classification on NIH
python src/zeroshot.py --checkpoint checkpoints/best.pt --prompt patient

# Retrieval + matched-vs-random diagnostic, either backbone
python3 scripts/stage3_medclip_diagnostic.py \
    --config configs/clipnorm.yaml --checkpoint checkpoints_clipnorm/best.pt
python3 scripts/stage3_medclip_diagnostic.py \
    --config configs/biomedclip_ft_lr1e5.yaml \
    --checkpoint checkpoints_biomedclip_lr1e5/best.pt \
    --out stage4_biomedclip_ft_diagnostic.md
```

## Fine-tuning BioMedCLIP (Stage 4)

```bash
python3 src/train.py --config configs/biomedclip_ft_lr1e5.yaml
```

The three `configs/biomedclip_ft_lr*.yaml` arms differ only in `lr_encoders`. ViT-B/16 is roughly 4x
the compute of ViT-B/32 per image, so these were run on a Kaggle T4 (~81 min per arm) via
[notebooks/kaggle_biomedclip/](notebooks/kaggle_biomedclip/), which also works with no Kaggle setup
beyond GPU + Internet — it clones this repo and pulls OpenI with kagglehub.

---

## References

- [OpenAI CLIP](https://arxiv.org/abs/2103.00020) — Radford et al., 2021
- [ConVIRT](https://arxiv.org/abs/2010.00747) — Zhang et al., 2020
- [CheXzero](https://www.nature.com/articles/s41551-022-00936-9) — Tiu et al., 2022
- [MedCLIP](https://arxiv.org/abs/2210.10163) — Wang et al., 2022
- [ClinicalBERT](https://arxiv.org/abs/1904.05342) — Alsentzer et al., 2019
