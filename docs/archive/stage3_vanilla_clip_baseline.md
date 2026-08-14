# Stage 3 Vanilla OpenAI CLIP Baseline

This run evaluates the original OpenAI CLIP dual tower without medical fine-tuning.

- Model: `ViT-B/32`
- Split: `test`
- Samples: `320`

## Retrieval Metrics

| Direction | R@1 | R@5 | R@10 | MedR |
|-----------|-----|-----|------|------|
| Image -> Text | 0.00 | 1.56 | 3.12 | 162.50 |
| Text -> Image | 0.31 | 2.81 | 4.06 | 166.00 |

## Matched vs Random Similarity

| Pair type | Mean | Std |
|-----------|------|-----|
| Matched image/report | 0.3043 | 0.0144 |
| Random mismatched pair | 0.3039 | 0.0144 |

Matched-minus-random gap: `0.0005`

Interpretation: this is the aligned general-domain CLIP baseline. It is useful because both towers were pretrained together, even though the pretraining data is not medical-domain specific.
