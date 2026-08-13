# Stage 3 BioMedCLIP Baseline

This run evaluates BioMedCLIP without additional fine-tuning on this project's OpenI split.

- Model: `hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224`
- Split: `test`
- Samples: `320`

## Retrieval Metrics

| Direction | R@1 | R@5 | R@10 | MedR |
|-----------|-----|-----|------|------|
| Image -> Text | 1.56 | 5.31 | 8.12 | 120.00 |
| Text -> Image | 1.25 | 6.25 | 8.75 | 108.50 |

## Matched vs Random Similarity

| Pair type | Mean | Std |
|-----------|------|-----|
| Matched image/report | 0.4095 | 0.0391 |
| Random mismatched pair | 0.3941 | 0.0484 |

Matched-minus-random gap: `0.0154`

Interpretation: this is the medical-domain pretrained dual-tower baseline. It tests how much alignment is available before project-specific OpenI fine-tuning.
