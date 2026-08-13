# Stage 3 Fine-Tuned MedCLIP Diagnostic

This run evaluates the project's fine-tuned MedCLIP checkpoint on OpenI retrieval.

- Checkpoint: `checkpoints/best.pt`
- Split: `test`
- Samples: `320`

## Retrieval Metrics

| Direction | R@1 | R@5 | R@10 | MedR |
|-----------|-----|-----|------|------|
| Image -> Text | 3.44 | 12.19 | 18.75 | 54.00 |
| Text -> Image | 3.75 | 11.88 | 18.44 | 52.50 |

## Matched vs Random Similarity

| Pair type | Mean | Std |
|-----------|------|-----|
| Matched image/report | 0.3037 | 0.0974 |
| Random mismatched pair | 0.1510 | 0.1439 |

Matched-minus-random gap: `0.1526`

Interpretation: a positive gap means paired X-ray/report examples are closer than random mismatches in the learned shared embedding space.
