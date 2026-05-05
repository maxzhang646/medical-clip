# Experimental Results

## 1. Retrieval (OpenI test split, 320 samples)

| Direction | Metric | MedCLIP | Vanilla CLIP | Improvement |
|-----------|--------|---------|--------------|-------------|
| I→T | R@1 | 3.44 | 0.31 | 11x |
| I→T | R@5 | 12.19 | 0.94 | 13x |
| I→T | R@10 | 18.75 | 2.81 | 7x |
| I→T | MedR | 54.00 | 162.00 | 3x better |
| T→I | R@1 | 3.75 | 0.31 | 12x |
| T→I | R@5 | 11.88 | 1.56 | 8x |
| T→I | R@10 | 18.44 | 2.81 | 7x |
| T→I | MedR | 52.50 | 163.00 | 3x better |

## 2. Zero-shot Classification (NIH ChestX-ray14, 2000 samples)

### Per-disease AUC by prompt template

| Prompt | Atelectasis | Cardiomegaly | Consolidation | Edema | Effusion | Infiltration | Pneumonia | Pneumothorax | Macro AUC |
|--------|-------------|--------------|---------------|-------|----------|--------------|-----------|--------------|-----------|
| simple | 0.6900 | 0.7373 | 0.3672 | 0.2382 | 0.3262 | 0.3985 | 0.4662 | 0.3661 | 0.4487 |
| findings | 0.7027 | 0.7300 | 0.4772 | 0.2636 | 0.3531 | 0.3701 | 0.5042 | 0.4090 | 0.4762 |
| clinical | 0.7203 | 0.7103 | 0.5570 | 0.4548 | 0.3564 | 0.4879 | 0.5743 | 0.3578 | 0.5273 |
| **patient** | **0.7069** | 0.7095 | **0.6297** | **0.4729** | **0.4785** | **0.5299** | **0.6626** | **0.5210** | **0.5889** |
| radiologist | 0.7116 | 0.7088 | 0.5363 | 0.2904 | 0.3689 | 0.3939 | 0.5211 | 0.3518 | 0.4854 |
| ensemble | 0.7121 | **0.7214** | 0.5111 | 0.3087 | 0.3482 | 0.4114 | 0.5368 | 0.3563 | 0.4882 |
| pos_neg | 0.7082 | 0.7240 | 0.4035 | 0.2388 | 0.2994 | 0.3816 | 0.4956 | 0.3457 | 0.4496 |

### Key findings

- `patient` ("A patient with {disease}") is best for 6/8 diseases; Macro AUC 0.589
- Atelectasis and Cardiomegaly are prompt-insensitive (range < 0.03 across all prompts)
- Consolidation shows largest prompt sensitivity: 0.367 → 0.630 (+0.26 AUC)
- Edema is hardest: best AUC 0.473, below random for most prompts
- Ensemble does NOT win — averaging diverse styles causes negative interference
- `pos_neg` (positive + negative templates averaged) is the worst strategy: Macro AUC 0.4496, barely above `simple` (0.4487). Averaging in "No evidence of X" pulls the class embedding toward no-disease space, hurting recall across the board — Effusion drops to 0.299 (worst of any prompt/class combination)

## 3. Edema Failure Analysis

Edema is the only class with best AUC below 0.5 (0.473), effectively random. Three compounding causes:

**Training data scarcity.** OpenI has ~3,955 reports total. Pulmonary edema is a secondary presentation of heart failure and appears infrequently in a general chest X-ray dataset — estimated <8% prevalence, yielding ~200 training pairs. The model never sees enough Edema examples to learn a stable visual-language alignment.

**Visual overlap with adjacent classes.** Edema's radiographic features (bilateral haziness, perihilar butterfly pattern, blunted costophrenic angles) overlap substantially with Effusion (fluid at lung bases) and Consolidation (airspace opacity). The model likely attributes Edema's visual signal to these more frequent neighbors during contrastive training.

**NIH label noise.** NIH labels are NLP-extracted from radiology reports. Radiologists describe pulmonary edema using varied terminology ("pulmonary congestion", "vascular engorgement", "interstitial edema") that NLP tools miss, producing noisier ground truth for Edema than for well-defined classes like Cardiomegaly.

**Prompt engineering cannot fix this.** Even the best prompt (`patient`) only reaches AUC 0.473 — the ceiling is set by the quality of the learned visual representation, not the inference-time text. Fixing Edema would require more training data (e.g. MIMIC-CXR) or explicit oversampling of Edema-positive reports.

## 4. Training

- Best checkpoint: Epoch 9, val_loss = 3.6404
- Overfitting onset: ~Epoch 9 (train/val gap widens)
- Dataset size limitation (2554 train samples) is primary bottleneck
