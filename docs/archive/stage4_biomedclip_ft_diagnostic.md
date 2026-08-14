# Stage 3 Fine-Tuned MedCLIP Diagnostic

This run evaluates the project's fine-tuned MedCLIP checkpoint on OpenI retrieval.

- Checkpoint: `checkpoints_biomedclip_lr3e6/best.pt`
- Split: `test`
- Samples: `320`

## Retrieval Metrics

| Direction | R@1 | R@5 | R@10 | MedR |
|-----------|-----|-----|------|------|
| Image -> Text | 5.00 | 17.50 | 25.31 | 46.50 |
| Text -> Image | 4.06 | 14.69 | 23.12 | 47.50 |

## Matched vs Random Similarity

| Pair type | Mean | Std |
|-----------|------|-----|
| Matched image/report | 0.4170 | 0.0177 |
| Random mismatched pair | 0.3932 | 0.0254 |

Matched-minus-random gap: `0.0238`
Normalized gap (gap / random std): `0.9379`
Per-query z-score, image->text: `1.0332`
Per-query z-score, text->image: `1.0323`

Interpretation: a positive gap means paired X-ray/report examples are closer than random mismatches in the learned shared embedding space.

Compare the raw gap only within one backbone: it scales with how tightly the embeddings are
packed, so a model whose embeddings sit in a narrow cone can rank better while showing a much
smaller absolute gap. Neither repair rescues cross-backbone comparison -- normalizing by the
spread, and even the per-query z-score, both rank BioMedCLIP below CLIP+ClinicalBERT while its
R@K is far higher. All three summarize central tendency, whereas ranking is a tail property:
what matters is how many distractors beat the true match, not the average distractor.

Use these statistics to answer 'is there any alignment at all' (vanilla CLIP: 0.0005 -> no).
For 'which model aligns better', MedR and R@K are the summary statistics, and they are already
rank-based by construction.
