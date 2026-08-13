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

Interpretation: a positive gap means paired X-ray/report examples are closer than random mismatches in the learned shared embedding space.
