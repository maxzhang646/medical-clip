# Stage 3 CLIP-Normalization Retraining Summary

This run retrained MedCLIP with CLIP image normalization instead of ImageNet normalization.

## Configuration

- Config: `configs/clipnorm.yaml`
- Checkpoint directory: `checkpoints_clipnorm`
- Best checkpoint: `checkpoints_clipnorm/best.pt`
- Epochs: `15`
- Batch size: `64`
- Image normalization: `clip`
- OpenI split files: `splits/openi_{train,val,test}_uids.txt`

## Training Curve

| Epoch | Train loss | Val loss | Best saved |
|-------|------------|----------|------------|
| 1 | 4.1610 | 4.1524 | yes |
| 2 | 4.0823 | 3.9356 | yes |
| 3 | 3.9182 | 3.8399 | yes |
| 4 | 3.7598 | 3.7889 | yes |
| 5 | 3.5855 | 3.7083 | yes |
| 6 | 3.4399 | 3.7063 | yes |
| 7 | 3.2873 | 3.6535 | yes |
| 8 | 3.1637 | 3.6804 | no |
| 9 | 3.0168 | 3.6426 | yes |
| 10 | 2.9387 | 3.6034 | yes |
| 11 | 2.8737 | 3.6198 | no |
| 12 | 2.7821 | 3.6409 | no |
| 13 | 2.7439 | 3.6115 | no |
| 14 | 2.7142 | 3.6113 | no |
| 15 | 2.6927 | 3.6106 | no |

Best validation loss: `3.6034` at epoch 10.

## Test Retrieval Diagnostic

Run:

```bash
python3 scripts/stage3_medclip_diagnostic.py \
  --config configs/clipnorm.yaml \
  --checkpoint checkpoints_clipnorm/best.pt \
  --indiana-dir /path/to/openi \
  --out stage3_medclip_diagnostic_clipnorm_retrained.md
```

Output:

| Direction | R@1 | R@5 | R@10 | MedR |
|-----------|-----|-----|------|------|
| Image -> Text | 3.75 | 12.81 | 20.62 | 49.50 |
| Text -> Image | 4.38 | 11.88 | 19.38 | 47.00 |

Matched-vs-random:

| Pair type | Mean | Std |
|-----------|------|-----|
| Matched image/report | 0.2804 | 0.1062 |
| Random mismatched pair | 0.1087 | 0.1566 |

Matched-minus-random gap: `0.1717`

## Interpretation

CLIP normalization did not clearly help when applied only at inference time to the old checkpoint. After retraining with CLIP normalization, the model improved the matched-minus-random gap and median rank. This suggests CLIP normalization is worth keeping as a serious training setting, even though the improvement is moderate rather than transformative.
