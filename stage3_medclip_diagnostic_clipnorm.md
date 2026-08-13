# Stage 3 Fine-Tuned MedCLIP Diagnostic

This run evaluates the project's fine-tuned MedCLIP checkpoint on OpenI retrieval.

- Checkpoint: `checkpoints/best.pt`
- Split: `test`
- Samples: `320`

## Retrieval Metrics

| Direction | R@1 | R@5 | R@10 | MedR |
|-----------|-----|-----|------|------|
| Image -> Text | 2.81 | 12.81 | 18.75 | 53.50 |
| Text -> Image | 3.44 | 10.94 | 18.75 | 49.50 |

## Matched vs Random Similarity

| Pair type | Mean | Std |
|-----------|------|-----|
| Matched image/report | 0.2975 | 0.0967 |
| Random mismatched pair | 0.1450 | 0.1431 |

Matched-minus-random gap: `0.1525`

Interpretation: a positive gap means paired X-ray/report examples are closer than random mismatches in the learned shared embedding space.
