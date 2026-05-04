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

### Key findings

- "patient" prompt ("A patient with {disease}") is best for 6/8 diseases
- Atelectasis and Cardiomegaly are prompt-insensitive (range < 0.03)
- Consolidation shows largest prompt sensitivity: 0.37 → 0.63 (+0.26)
- Edema is hardest: best AUC 0.47, below random for most prompts
- Ensemble does NOT win — negative interference between prompt styles

## 3. Training

- Best checkpoint: Epoch 9, val_loss = 3.6404
- Overfitting onset: ~Epoch 9 (train/val gap widens)
- Dataset size limitation (2554 train samples) is primary bottleneck
