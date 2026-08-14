# Paired checkpoint comparison (test split)

- A: `checkpoints_clipnorm/best.pt` (CLIP+ClinicalBERT)
- B: `checkpoints_biomedclip_lr1e5_ep5/best.pt` (BioMedCLIP FT)

## Image -> Text

| metric | value |
|---|---|
| queries | 320 |
| improved under BioMedCLIP FT | 160 (50.0%) |
| degraded under BioMedCLIP FT | 155 (48.4%) |
| unchanged | 5 |
| median rank, CLIP+ClinicalBERT | 49.5 |
| median rank, BioMedCLIP FT | 45.5 |
| median rank change | -0.5 |
| sign test z | +0.28 |
| R@1: 3.75 -> 5.94, gained 16 / lost 9, McNemar chi2 1.44 |
| R@5: 12.81 -> 16.56, gained 42 / lost 30, McNemar chi2 1.68 |
| R@10: 20.62 -> 26.25, gained 58 / lost 40, McNemar chi2 2.95 |

### Largest rank improvements under BioMedCLIP FT

| query | uid | rank A | rank B | change |
|---|---|---|---|---|
| 110 | 1555 | 302 | 99 | -203 |
| 171 | 2158 | 200 | 1 | -199 |
| 209 | 2685 | 217 | 38 | -179 |
| 133 | 1824 | 209 | 31 | -178 |
| 78 | 1168 | 215 | 46 | -169 |

### Largest rank degradations under BioMedCLIP FT

| query | uid | rank A | rank B | change |
|---|---|---|---|---|
| 141 | 1876 | 9 | 271 | +262 |
| 118 | 1630 | 12 | 262 | +250 |
| 56 | 828 | 48 | 282 | +234 |
| 98 | 1425 | 62 | 288 | +226 |
| 76 | 1072 | 64 | 276 | +212 |

## Text -> Image

| metric | value |
|---|---|
| queries | 320 |
| improved under BioMedCLIP FT | 152 (47.5%) |
| degraded under BioMedCLIP FT | 165 (51.6%) |
| unchanged | 3 |
| median rank, CLIP+ClinicalBERT | 47.0 |
| median rank, BioMedCLIP FT | 46.0 |
| median rank change | +2.0 |
| sign test z | -0.73 |
| R@1: 4.38 -> 4.06, gained 12 / lost 13, McNemar chi2 0.00 |
| R@5: 11.88 -> 15.62, gained 40 / lost 28, McNemar chi2 1.78 |
| R@10: 19.38 -> 24.38, gained 59 / lost 43, McNemar chi2 2.21 |

### Largest rank improvements under BioMedCLIP FT

| query | uid | rank A | rank B | change |
|---|---|---|---|---|
| 9 | 181 | 237 | 9 | -228 |
| 110 | 1555 | 293 | 82 | -211 |
| 38 | 572 | 220 | 22 | -198 |
| 293 | 3694 | 309 | 116 | -193 |
| 133 | 1824 | 226 | 33 | -193 |

### Largest rank degradations under BioMedCLIP FT

| query | uid | rank A | rank B | change |
|---|---|---|---|---|
| 173 | 2186 | 2 | 246 | +244 |
| 242 | 3084 | 26 | 256 | +230 |
| 256 | 3275 | 11 | 229 | +218 |
| 253 | 3229 | 54 | 249 | +195 |
| 219 | 2784 | 10 | 199 | +189 |

