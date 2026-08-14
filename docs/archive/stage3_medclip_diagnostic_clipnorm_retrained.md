# Stage 3 Fine-Tuned MedCLIP Diagnostic

This run evaluates the project's fine-tuned MedCLIP checkpoint on OpenI retrieval.

- Checkpoint: `checkpoints_clipnorm/best.pt`
- Split: `test`
- Samples: `320`

## Retrieval Metrics

| Direction | R@1 | R@5 | R@10 | MedR |
|-----------|-----|-----|------|------|
| Image -> Text | 3.75 | 12.81 | 20.62 | 49.50 |
| Text -> Image | 4.38 | 11.88 | 19.38 | 47.00 |

## Matched vs Random Similarity

| Pair type | Mean | Std |
|-----------|------|-----|
| Matched image/report | 0.2804 | 0.1062 |
| Random mismatched pair | 0.1087 | 0.1566 |

Matched-minus-random gap: `0.1717`
Normalized gap (gap / random std): `1.0965`
Per-query z-score, image->text: `1.0740`
Per-query z-score, text->image: `1.0543`

Interpretation: a positive gap means paired X-ray/report examples are closer than random mismatches in the learned shared embedding space.

Compare the raw gap only within one backbone: it scales with how tightly the embeddings are
packed, so a model whose embeddings sit in a narrow cone can rank better while showing a much
smaller absolute gap. Normalizing by the spread fixes the scale but not the deeper problem --
both are global mean shifts, while retrieval is decided per query. The per-query z-score is the
statistic that tracks ranking; R@K and MedR remain the authority.
