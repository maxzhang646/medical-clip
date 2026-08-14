# Do near-duplicate reports explain retrieval failure? (test split)

- Checkpoint: `checkpoints_biomedclip_lr1e5_ep5/best.pt`
- Near-duplicate threshold: TF-IDF cosine >= 0.8
- Queries: 320

## Rank by number of near-duplicate competitors

| near-duplicates | queries | median rank | R@5 |
|---|---|---|---|
| 0 | 236 | 37.5 | 19.9% |
| 1-2 | 39 | 65.0 | 7.7% |
| 3-5 | 35 | 79.0 | 2.9% |
| 6+ | 10 | 60.0 | 20.0% |

Correlation between near-duplicate count and rank: `+0.040`

## Are failures wrong, or just ambiguous?

| group | queries | share |
|---|---|---|
| retrieved the true report in top-5 | 53 | 16.6% |
| failed, but rank-1 report is a near-duplicate of the true one | 0 | 0.0% |
| failed, rank-1 report is genuinely different | 267 | 83.4% |

Median TF-IDF similarity between the rank-1 report and the true report, over failures: `0.105`

## Coarse semantic agreement (normal vs abnormal)

- Reports the regex calls normal: 185 / 320 (57.8%)
- Rank-1 shares the true report's normal/abnormal label: 57.8% overall, 55.4% among failures
- Chance agreement given the base rate: 51.2%

TF-IDF similarity is lexical: two normal reports worded differently score low even
though they say the same thing. This label is the crude semantic backstop.

## Failure examples with the highest text similarity to the true report

### query 12 (uid 229) — true rank 6, text similarity to rank-1 0.605

**True report:** Heart size within normal limits. No focal alveolar consolidation, no definite pleural effusion seen. No typical findings of pulmonary edema. Mediastinal calcification and dense right upper lung nodule suggest a previous granulomatous process. No acute cardiopulmonary findings

**Rank-1 retrieved:** Heart size within normal limits, stable mediastinal and hilar contours. No focal alveolar consolidation, no definite pleural effusion seen. No typical findings of pulmonary edema. No acute findings

### query 71 (uid 1044) — true rank 8, text similarity to rank-1 0.493

**True report:** The heart size and pulmonary vascularity appear within normal limits. There has been clearing of left base airspace opacities. The lungs now appear clear. No pneumothorax or pleural effusion is seen. The lungs appear hyperexpanded consistent with emphysema. 1. Hyperexpanded lungs consistent with emphysema. 2. No evidence of acute disease.

**Rank-1 retrieved:** The lungs are hyperexpanded consistent with emphysema. The heart size and pulmonary vascularity appear within normal limits. No pneumothorax or pleural effusion is seen. Patchy airspace disease is present in the right middle lobe. Degenerative changes are present spine. 1. Hyperexpanded lungs suggesting emphysema. 2. Patchy right middle lobe airspace disease. May represent pneumonia. Followup examination is suggested following treatment to confirm clearing of the opacities. A 4 to 6 XXXX post treatment interval film would be reasonable to allow clearing of inflammatory opacities.

### query 182 (uid 2297) — true rank 38, text similarity to rank-1 0.444

**True report:** The cardiac contours are normal. The lungs are clear. Thoracic spondylosis. Mild XXXX XXXX curvature thoracolumbar junction. No active pulmonary disease.

**Rank-1 retrieved:** Cardiac and mediastinal contours are within normal limits. Prior granulomatous disease. The lungs are clear. Thoracic spondylosis. No acute findings.

### query 263 (uid 3373) — true rank 13, text similarity to rank-1 0.372

**True report:** The heart is normal size. The mediastinum is unremarkable. There is no pleural effusion, pneumothorax, or focal airspace disease. The XXXX are unremarkable. No acute cardiopulmonary abnormality.

**Rank-1 retrieved:** Lungs are clear bilaterally with no focal infiltrate, pleural effusion, or pneumothoraces. Cardiomediastinal silhouette is within normal limits. XXXX and soft tissues are unremarkable. No acute cardiopulmonary abnormality. .

### query 179 (uid 2246) — true rank 67, text similarity to rank-1 0.366

**True report:** Normal heart size and mediastinal contour. Right lung base airspace disease on frontal XXXX. XXXX opacities in the left lung base consistent with atelectasis. No pneumothorax. No pleural effusion. Mild wedge XXXX deformity of T12. Right lung base airspace disease and left base atelectasis.

**Rank-1 retrieved:** Stable cardiomediastinal silhouette. Stable XXXX opacity in the left base, XXXX scarring or atelectasis. Rounded calcified density in the left lung base, XXXX calcified granuloma. No XXXX consolidation. No pleural effusion or pneumothorax. Stable degenerative changes of the spine. No acute cardiopulmonary abnormality.

### query 266 (uid 3421) — true rank 95, text similarity to rank-1 0.351

**True report:** PA and lateral views of the chest were obtained. The cardiomediastinal silhouette is normal in size and configuration. Mildly tortuous thoracic aorta. The lungs are well aerated. There is no pneumothorax, pleural effusion, or focal air space consolidation. Mild elevation right hemidiaphragm. 1. No acute cardiopulmonary disease.

**Rank-1 retrieved:** Heart size and pulmonary vascularity normal. The stomach contour normal. There is right hemidiaphragm elevation. Lungs are clear. Degenerative changes in the thoracic spine. Right hemidiaphragm elevation. No acute cardiopulmonary process.

