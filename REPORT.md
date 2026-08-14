# Medical CLIP on chest X-rays: what held up and what did not

A CLIP-style model trained on 2,554 chest X-ray / radiology-report pairs from Indiana University
OpenI, evaluated by image-text retrieval on OpenI and by zero-shot classification on NIH
ChestX-ray14. This report consolidates six stages of experiments into one narrative.

Several claims made earlier in the project were later overturned by its own experiments. Those
reversals are kept here rather than edited away — they are the part of the work with transferable
content, and each one names the flaw that produced it. Per-stage documents are archived under
[`docs/archive/`](docs/archive/).

---

## 1. The question

Two encoders from unrelated pretraining runs — OpenAI CLIP's ViT-B/32 image tower and ClinicalBERT —
are pushed into one embedding space by contrastive fine-tuning. Does that actually produce medical
image-text alignment, and how would we know?

Everything downstream is an attempt to answer the second half of that question honestly.

## 2. Setup

| | |
|---|---|
| Training data | OpenI, 2,554 image/report pairs (Findings + Impression as the caption) |
| Splits | Patient-level, fixed in [`splits/`](splits/): 2,554 / 319 / 320 |
| Retrieval eval | OpenI test split, 320 pairs, R@K and median rank |
| Zero-shot eval | NIH ChestX-ray14, 2,000 images (seed-42 subset), 8 classes, AUC-ROC |
| Backbone A | CLIP ViT-B/32 + ClinicalBERT, projections trained from scratch |
| Backbone B | BioMedCLIP (ViT-B/16 + PubMedBERT), fine-tuned end to end |
| Loss | Symmetric InfoNCE, batch 64 |

Backbone B adds no new projection layers on purpose: BioMedCLIP ships trained projection heads, and a
randomly initialised Linear on top would destroy the pretrained alignment and invalidate the
comparison. It also keeps its own `logit_scale` (τ ≈ 0.0117) rather than the configured 0.07.

## 3. Results

### 3.1 Retrieval (OpenI test, 320 candidates)

| Model | I→T R@1 | R@5 | R@10 | MedR | T→I R@5 | MedR |
|-------|---------|-----|------|------|---------|------|
| Vanilla OpenAI CLIP (zero-shot) | 0.00 | 1.56 | 3.12 | 162.5 | 2.81 | 166.0 |
| BioMedCLIP (zero-shot) | 1.56 | 5.31 | 8.12 | 120.0 | 6.25 | 108.5 |
| CLIP+ClinicalBERT (fine-tuned) | 3.75 | 12.81 | 20.62 | 49.5 | 11.88 | 47.0 |
| BioMedCLIP (fine-tuned, lr 1e-5) | 6.25 | 17.50 | 27.19 | 45.5 | 17.50 | 46.0 |

Random baseline at 320 candidates is R@5 = 1.56%. Vanilla CLIP sits exactly there: general-domain
image-text alignment does not transfer to chest X-rays and their reports at all. Fine-tuning is what
creates alignment.

Filling in the 2×2 of initialization × fine-tuning (I→T R@5):

|  | zero-shot | fine-tuned |
|--|-----------|------------|
| CLIP + ClinicalBERT | 1.56 | 12.81 |
| BioMedCLIP | 5.31 | **17.50** |

Both main effects are positive and the combination is best. **But see §4.1 — this table does not
support the conclusion it appears to support.**

### 3.2 Zero-shot classification (NIH, 2,000 images)

| Prompt | CLIP+ClinicalBERT | BioMedCLIP FT | Δ |
|--------|-------------------|---------------|---|
| simple | 0.4487 | 0.6153 | +0.167 |
| findings | 0.4762 | 0.6719 | +0.196 |
| **clinical** | 0.5273 | **0.6824** | +0.155 |
| patient | **0.5889** | 0.6642 | +0.075 |
| radiologist | 0.4854 | 0.6542 | +0.169 |
| ensemble | 0.4882 | 0.6701 | +0.182 |
| pos_neg | 0.4496 | 0.5913 | +0.142 |

All seven templates improve. This is the strongest evidence in the project, and it is on the harder
test — cross-dataset transfer rather than in-domain retrieval.

For context, at 0.682 Macro AUC the model is within 0.05 of MedCLIP (Wang et al., 2022, ~200K pairs)
using 78× less data, and reaches 78% of CheXzero's AUC (Tiu et al., 2022, ~227K pairs) with 89× less.

### 3.3 Training behaviour

| Model | Best val_loss | Best epoch |
|-------|---------------|------------|
| CLIP+ClinicalBERT | 3.6404 | 9 |
| BioMedCLIP, 15-epoch schedule | 3.6278 | 3 |
| BioMedCLIP, 5-epoch schedule | 3.6504 | 3 |

An already-aligned model peaks at epoch 3 under two different cosine schedules, so the ceiling is set
by the data, not the schedule: roughly three passes exhaust what 2,554 pairs can teach it. The
15-epoch run ends at val_loss 4.2297, above the ln(64) = 4.159 random baseline — the shorter budget
is not better, it is safer, and it costs a third of the compute (26 vs 81 minutes on a T4).

Learning rate between 1e-5 and 3e-6 makes no measurable difference; 1e-6 underfits (R@5 12.50).
The catastrophic forgetting anticipated for the highest LR never appeared.

## 4. Claims that did not survive

### 4.1 "Fine-tuned BioMedCLIP retrieves better" — not established

R@5 goes 12.81 → 17.50, which reads as a decisive win. A paired McNemar test on the top-K indicator —
the matched test for a threshold metric — says otherwise:

| Direction | Metric | gained | lost | χ² | p |
|-----------|--------|--------|------|----|---|
| I→T | R@1 | 16 | 9 | 1.44 | 0.23 |
| I→T | R@5 | 42 | 30 | 1.68 | 0.20 |
| I→T | R@10 | 58 | 40 | 2.95 | 0.09 |
| T→I | R@5 | 40 | 28 | 1.78 | 0.18 |

Gaining 42 top-5 hits while losing 30 is a net +12 against a lot of churn. **320 queries cannot
resolve a 4-point R@5 difference.** Across all 320 queries, ranks improve for 160 and degrade for
155 (sign test z = +0.28).

An earlier draft claimed this comparison was "~2.2 SE, so this one holds". That came from an
independent-sample standard error plus an assumption that pairing would only strengthen it. Running
the paired test was the whole fix.

The backbone claim still stands, but on §3.2, not §3.1.

### 4.2 "`patient` is the best prompt" — false

Re-running the ablation on a second backbone is what exposed this. `clinical` wins there and
`patient` drops to 4th. Which template is best is a property of a checkpoint, not of prompt design.

The negative results held up better:

| Claim | Verdict |
|---|---|
| `patient` is best | **False** — checkpoint-specific |
| `ensemble` loses to the best single prompt | **True on both**, but the deficit collapses from −0.101 to −0.012 |
| `pos_neg` is worst | **True on both** |

The ensemble result is more interesting as a margin than as a verdict. On the weaker backbone,
mixing prompt styles cost a tenth of an AUC point; on the stronger one the penalty is noise. So
"negative interference between prompt styles" is not a property of ensembling — it is a symptom of a
weak visual representation. The same thing shows up as overall prompt sensitivity shrinking (spread
across templates 0.140 → 0.091): **prompt engineering matters most when the visual encoder is weak.**

### 4.3 "Edema fails because OpenI under-represents it" — wrong cause

Edema was the worst class on the first backbone (best AUC 0.473, below random), attributed to
training scarcity, visual overlap with Effusion and Consolidation, and NIH label noise, with the
conclusion that fixing it needed MIMIC-CXR.

On the same 2,554 pairs, fine-tuned BioMedCLIP reaches **Edema 0.788** and Effusion 0.782. The
diagnosis that the ceiling was the visual representation was right; the prescription was too narrow.
Training-data scarcity and inter-class overlap cannot be the binding constraints, because the same
data and the same overlapping classes now work.

Infiltration replaces Edema as the one systematically broken class: 0.374–0.479 across all seven
templates, below random everywhere, on ~340 positives — too many for that to be noise. This is the
project's clearest open question.

### 4.4 "Boilerplate reports make exact matching impossible" — false

| near-duplicate reports (TF-IDF ≥ 0.8) | queries | median rank | R@5 |
|---|---|---|---|
| 0 | 236 | 37.5 | 19.9% |
| 1–2 | 39 | 65.0 | 7.7% |
| 3–5 | 35 | 79.0 | 2.9% |

Duplicates hurt where they exist, but 236 of 320 queries have none and still reach only 19.9% R@5.
Correlation between duplicate count and rank is +0.04, stable across thresholds 0.5–0.8.

### 4.5 "Failed retrievals are still medically close" — false

Among the 267 failures: median TF-IDF similarity between the rank-1 and the true report is **0.105**;
zero are near-duplicates; and rank-1 agrees with the truth on a coarse normal/abnormal label 55.4% of
the time against 51.2% expected by chance — 1.4 SE, not significant.

TF-IDF is lexical and would undercount two normal reports worded differently, which is why the
semantic label is there as a backstop. It agrees. When this model misses, the report it ranks first
is not reliably even in the right normal/abnormal category.

Exact-match retrieval is therefore not being unfairly strict. The failures are failures.

## 5. Methodological notes

Four traps hit during this work, each caught by a measurement rather than by inspection:

**Comparing a fine-tuned model against a zero-shot one.** Stage 3 concluded that task-specific
fine-tuning mattered more than medical pretraining, from a table where only one side had been
trained. Fine-tuning BioMedCLIP on the same data reverses the ranking.

**Selecting example cases by one model's performance.** The first qualitative comparison chose cases
from the CLIP+ClinicalBERT model's own hits and misses, then ran both models on them. BioMedCLIP
looked much worse on the first model's wins and much better on its losses — pure regression to the
mean. Case selection has to be independent of both models, or cover all of them.

**Reading a threshold metric with a mean-shift statistic.** The matched-minus-random similarity gap
ranks BioMedCLIP below CLIP+ClinicalBERT (0.023 vs 0.172) while its R@K is higher, because
BioMedCLIP's embeddings sit in a narrow cone — unrelated pairs already score 0.394 cosine. Two
repairs were tried and both failed: normalizing by the spread (0.938 vs 1.097) and a per-query
z-score (1.033 vs 1.074) preserve the wrong order. All three summarize central tendency, while
ranking is a tail property. The gap family can answer "is there any alignment at all" (vanilla CLIP:
0.0005 → no) and support within-backbone ablations; it cannot rank models.

**Two implementations of one metric.** `torch.argsort` and `np.argsort(-x)` break float ties
differently, which was enough to make two scripts in this repo report 12.81% and 13.12% R@5 for the
same checkpoint. Both now rank in numpy.

## 6. Limitations

- **The retrieval test set is underpowered.** 320 queries cannot resolve differences of a few points
  in R@K. Every comparison in §3.1 between the fine-tuned models is inside one standard error.
- **The backbone comparison has an uncontrolled variable.** BioMedCLIP is ViT-B/16 against our
  ViT-B/32, so part of the gain is patch size, not medical pretraining. BioMedCLIP publishes no B/32
  variant, so this cannot be separated here.
- **Per-class NIH AUCs for rare findings are noisy**: roughly ±0.06 for Edema (~40 positives) and
  ±0.09 for Pneumonia (~26). Single-class differences under ~0.1 should not be read as real.
- 2,554 training pairs; no cross-attention, no hard-negative mining, no MIMIC-CXR.
- NIH labels are NLP-extracted from reports and known to be noisy, Infiltration especially.
- The project title mentions evidence retrieval; that application layer is planned but not built.

This is a study of whether medical image-text alignment is learned and how to measure it, not a
clinical diagnostic model.

## 7. What follows from this

1. **Infiltration below random on every prompt** is the one unexplained result. Its NIH labels overlap
   heavily with Consolidation in definition, so the first step is the label co-occurrence structure,
   not the model.
2. **A larger retrieval evaluation** — the current one cannot answer the question it is being asked.
   Evaluating against the full OpenI set rather than the 320-pair test split would give more power at
   no training cost.
3. **MIMIC-CXR** if the goal shifts from measurement to performance: 227K pairs against 2,554, and
   §3.3 shows the current data is exhausted after three epochs.

## 8. Reproducing

```bash
# Fine-tune either backbone
python3 src/train.py --config configs/clipnorm.yaml --indiana-dir /path/to/openi
python3 src/train.py --config configs/biomedclip_ft_lr1e5_ep5.yaml --indiana-dir /path/to/openi

# Retrieval + matched-vs-random diagnostics
python3 scripts/stage3_medclip_diagnostic.py --config configs/biomedclip_ft_lr1e5_ep5.yaml \
    --checkpoint checkpoints_biomedclip_lr1e5_ep5/best.pt

# Zero-shot prompt ablation (NIH; needs the 45GB dataset, run on Kaggle)
python3 src/zeroshot.py --config configs/biomedclip_ft_lr1e5_ep5.yaml \
    --checkpoint checkpoints_biomedclip_lr1e5_ep5/best.pt --prompt all --sample 2000

# Paired comparison of two checkpoints, and the failure analysis
python3 scripts/stage6_compare_checkpoints.py --config-a ... --config-b ...
python3 scripts/stage6_failure_analysis.py --config ... --checkpoint ...
```

Kaggle notebooks for the GPU runs are in [`notebooks/`](notebooks/). NIH zero-shot needs the dataset
attached as an input; OpenI falls back to kagglehub.

## References

- [CLIP](https://arxiv.org/abs/2103.00020) — Radford et al., 2021
- [ConVIRT](https://arxiv.org/abs/2010.00747) — Zhang et al., 2020 — the closest predecessor: chest
  X-ray / report contrastive pretraining
- [CheXzero](https://www.nature.com/articles/s41551-022-00936-9) — Tiu et al., 2022
- [MedCLIP](https://arxiv.org/abs/2210.10163) — Wang et al., 2022 — note that this project borrows the
  name but not the method; there is no CheXpert-labeler soft-label matrix here, so it is closer to
  ConVIRT
- [BioMedCLIP](https://arxiv.org/abs/2303.00915) — Zhang et al., 2023
- [ClinicalBERT](https://arxiv.org/abs/1904.05342) — Alsentzer et al., 2019
- OpenI — Demner-Fushman et al., JAMIA 2016 · NIH ChestX-ray14 — Wang et al., CVPR 2017
