# Stage 3 Notes: Strengthening Multimodal Evaluation

## Goal

Stage 3 moves beyond "the model runs" and asks:

```text
What did the image-text embedding space actually learn?
```

The primary evaluation remains bidirectional retrieval:

- image-to-text retrieval
- text-to-image retrieval
- Recall@K
- median rank

But numeric retrieval metrics are not enough. We also need qualitative examples to inspect whether retrieved reports are medically similar, even when the exact paired report is not retrieved.

## Qualitative Retrieval Examples

Generate OpenI image-to-report retrieval examples with:

```bash
python3 scripts/stage3_retrieval_examples.py \
  --direction image-to-text \
  --checkpoint checkpoints/best.pt \
  --top-k 5 \
  --num-queries 8
```

Generate report-to-image retrieval examples with:

```bash
python3 scripts/stage3_retrieval_examples.py \
  --direction text-to-image \
  --checkpoint checkpoints/best.pt \
  --top-k 5 \
  --num-queries 8 \
  --out stage3_text_to_image_examples.md
```

If OpenI is outside `data/indiana`, pass:

```bash
python3 scripts/stage3_retrieval_examples.py \
  --direction image-to-text \
  --checkpoint checkpoints/best.pt \
  --indiana-dir /path/to/openi \
  --top-k 5 \
  --num-queries 8
```

The script writes:

```text
stage3_retrieval_examples.md
stage3_text_to_image_examples.md
```

Each case includes:

- query UID
- image path
- ground-truth report
- ground-truth full rank
- top-k retrieved report UIDs
- similarity scores
- retrieved report text
- whether each retrieved item is the exact matched report

## Current Run

The current qualitative run used:

```bash
python3 scripts/stage3_retrieval_examples.py \
  --direction image-to-text \
  --checkpoint checkpoints/best.pt \
  --indiana-dir /Users/wenluzhang/.cache/kagglehub/datasets/raddar/chest-xrays-indiana-university/versions/2 \
  --num-queries 8 \
  --top-k 5
```

It produced:

```text
Image-to-text Recall@5: 12.50%
Output: stage3_retrieval_examples.md
```

The current report-to-image run used:

```bash
python3 scripts/stage3_retrieval_examples.py \
  --direction text-to-image \
  --checkpoint checkpoints/best.pt \
  --indiana-dir /Users/wenluzhang/.cache/kagglehub/datasets/raddar/chest-xrays-indiana-university/versions/2 \
  --num-queries 8 \
  --top-k 5 \
  --out stage3_text_to_image_examples.md
```

It produced:

```text
Text-to-image Recall@5: 11.87%
Output: stage3_text_to_image_examples.md
```

The script selects a mixed set of examples by default:

- some where the exact ground-truth report is retrieved in top-k
- some where the exact ground-truth report is missed

This is useful because Stage 3 is about understanding both success and failure modes.

## How To Read The Examples

For each case, first compare the query ground-truth report with the retrieved reports.

Ask:

1. Did the model retrieve the exact paired report?
2. If not, did it retrieve reports with similar medical content?
3. Are the top reports similar because of real findings, or because they share generic phrases like "no acute cardiopulmonary disease"?
4. Are failures caused by visually subtle findings, rare abnormalities, or report boilerplate?
5. Do repeated normal reports dominate the retrieval results?

## Early Observations

The first qualitative run shows a realistic pattern:

- The exact paired report is often not rank 1.
- Some misses are still semantically close, especially for normal or low-volume chest reports.
- Many normal reports share near-identical phrasing, which can make exact report retrieval hard.
- Retrieval quality should be judged both by exact match rank and by clinical/report similarity.

This matters because the project is learning multimodal alignment, not memorizing report IDs.

## Embedding Visualization

Generate a t-SNE visualization of image and report embeddings with:

```bash
env MPLCONFIGDIR=/private/tmp/xray_mpl_config python3 scripts/stage3_embedding_viz.py \
  --checkpoint checkpoints/best.pt \
  --indiana-dir /path/to/openi \
  --split test \
  --max-samples 320 \
  --out figures/stage3_embedding_tsne.png
```

Current output:

```text
figures/stage3_embedding_tsne.png
```

The plot contains both modalities:

- circle markers: image embeddings
- x markers: report embeddings

Colors are assigned from simple report keyword groups such as `No acute`, `Normal`, `Effusion`, `Atelectasis`, and `Other`. These are weak heuristic labels, not ground-truth disease labels.

Current observation:

- Image and report embeddings still form visibly different modality regions in t-SNE.
- This suggests the shared embedding space is only partially aligned.
- That pattern is consistent with the modest retrieval scores, where Recall@5 is around 12%.
- The visualization should be treated as diagnostic evidence, not as a formal metric.

## Vanilla OpenAI CLIP Baseline

Run the original OpenAI CLIP dual tower on the same OpenI split with:

```bash
python3 scripts/stage3_vanilla_clip_baseline.py \
  --indiana-dir /path/to/openi \
  --out stage3_vanilla_clip_baseline.md
```

This baseline is important because both towers are already aligned by OpenAI CLIP pretraining:

```text
OpenAI CLIP image encoder + OpenAI CLIP text encoder
```

But it is not medical-domain trained, so it may not understand radiology reports well.

Current run:

```bash
python3 scripts/stage3_vanilla_clip_baseline.py \
  --indiana-dir /Users/wenluzhang/.cache/kagglehub/datasets/raddar/chest-xrays-indiana-university/versions/2 \
  --out stage3_vanilla_clip_baseline.md
```

Output:

```text
Image-to-text: R@1 0.00, R@5 1.56, R@10 3.12, MedR 162.50
Text-to-image: R@1 0.31, R@5 2.81, R@10 4.06, MedR 166.00
Matched-minus-random similarity gap: 0.0005
```

Interpretation:

- Vanilla CLIP's two towers are aligned in a general image-text space.
- That alignment does not transfer well to exact OpenI X-ray/report retrieval.
- The near-zero matched-minus-random gap means matched image/report pairs are barely more similar than random mismatched pairs.
- This gives a useful baseline for judging whether the current MedCLIP fine-tuning actually learns domain-specific alignment.

## Fine-Tuned MedCLIP Diagnostic

Run the same diagnostic for the fine-tuned project checkpoint with:

```bash
python3 scripts/stage3_medclip_diagnostic.py \
  --checkpoint checkpoints/best.pt \
  --indiana-dir /path/to/openi \
  --out stage3_medclip_diagnostic.md
```

Current run:

```bash
python3 scripts/stage3_medclip_diagnostic.py \
  --checkpoint checkpoints/best.pt \
  --indiana-dir /Users/wenluzhang/.cache/kagglehub/datasets/raddar/chest-xrays-indiana-university/versions/2 \
  --out stage3_medclip_diagnostic.md
```

Output:

```text
Image-to-text: R@1 3.44, R@5 12.19, R@10 18.75, MedR 54.00
Text-to-image: R@1 3.75, R@5 11.88, R@10 18.44, MedR 52.50
Matched-minus-random similarity gap: 0.1526
```

Direct comparison:

| Model | Matched mean | Random mean | Gap |
|-------|--------------|-------------|-----|
| Vanilla OpenAI CLIP | 0.3043 | 0.3039 | 0.0005 |
| Fine-tuned MedCLIP | 0.3037 | 0.1510 | 0.1526 |

Interpretation:

- Vanilla CLIP assigns almost the same similarity to matched and random OpenI pairs.
- Fine-tuned MedCLIP keeps matched pairs around the same similarity level but pushes random mismatches much lower.
- This is evidence that project fine-tuning did learn domain-specific image/report alignment.
- The remaining issue is ranking strength: the gap is real, but not large enough to make exact retrieval consistently rank 1.

## Preprocessing Ablation

The project now supports two image normalization modes in `configs/base.yaml`:

```yaml
data:
  image_normalization: imagenet  # options: imagenet, clip
```

Run an inference-only ablation with:

```bash
python3 scripts/stage3_medclip_diagnostic.py \
  --checkpoint checkpoints/best.pt \
  --indiana-dir /path/to/openi \
  --image-normalization imagenet \
  --out stage3_medclip_diagnostic_imagenet.md

python3 scripts/stage3_medclip_diagnostic.py \
  --checkpoint checkpoints/best.pt \
  --indiana-dir /path/to/openi \
  --image-normalization clip \
  --out stage3_medclip_diagnostic_clipnorm.md
```

Current inference-only result:

| Normalization | I->T R@1 | I->T R@5 | I->T R@10 | I->T MedR | T->I R@1 | T->I R@5 | T->I R@10 | T->I MedR | Gap |
|---------------|----------|----------|-----------|-----------|----------|----------|-----------|-----------|-----|
| ImageNet | 3.44 | 12.19 | 18.75 | 54.00 | 3.75 | 11.88 | 18.44 | 52.50 | 0.1526 |
| CLIP | 2.81 | 12.81 | 18.75 | 53.50 | 3.44 | 10.94 | 18.75 | 49.50 | 0.1525 |

Interpretation:

- Swapping to CLIP normalization at inference time does not clearly improve the current checkpoint.
- The matched-minus-random gap is essentially unchanged.
- This is not a full training ablation because `checkpoints/best.pt` was trained with the original ImageNet normalization.
- A stronger test would retrain from scratch with `image_normalization: clip`.

That stronger test has now been run with:

```bash
python3 src/train.py \
  --config configs/clipnorm.yaml \
  --indiana-dir /Users/wenluzhang/.cache/kagglehub/datasets/raddar/chest-xrays-indiana-university/versions/2
```

Training summary:

- Config: `configs/clipnorm.yaml`
- Best checkpoint: `checkpoints_clipnorm/best.pt`
- Best epoch: 10
- Best validation loss: 3.6034
- Full training record: `stage3_clipnorm_training_summary.md`

Test diagnostic:

```bash
python3 scripts/stage3_medclip_diagnostic.py \
  --config configs/clipnorm.yaml \
  --checkpoint checkpoints_clipnorm/best.pt \
  --indiana-dir /path/to/openi \
  --out stage3_medclip_diagnostic_clipnorm_retrained.md
```

| Model | I->T R@1 | I->T R@5 | I->T R@10 | I->T MedR | T->I R@1 | T->I R@5 | T->I R@10 | T->I MedR | Gap |
|-------|----------|----------|-----------|-----------|----------|----------|-----------|-----------|-----|
| Original ImageNet-norm checkpoint | 3.44 | 12.19 | 18.75 | 54.00 | 3.75 | 11.88 | 18.44 | 52.50 | 0.1526 |
| Retrained CLIP-norm checkpoint | 3.75 | 12.81 | 20.62 | 49.50 | 4.38 | 11.88 | 19.38 | 47.00 | 0.1717 |

Interpretation:

- CLIP normalization is not useful as a simple inference-time swap on the old checkpoint.
- When the model is retrained with CLIP normalization, retrieval becomes moderately better.
- The improvement is clearest in R@10, median rank, and matched-minus-random gap.
- R@5 remains similar, so preprocessing is not the only limiting factor.

## BioMedCLIP Baseline

Run the medical pretrained BioMedCLIP baseline with:

```bash
python3 scripts/stage3_biomedclip_baseline.py \
  --indiana-dir /path/to/openi \
  --out stage3_biomedclip_baseline.md
```

Current run:

```bash
python3 scripts/stage3_biomedclip_baseline.py \
  --indiana-dir /Users/wenluzhang/.cache/kagglehub/datasets/raddar/chest-xrays-indiana-university/versions/2 \
  --out stage3_biomedclip_baseline.md
```

Output:

```text
Image-to-text: R@1 1.56, R@5 5.31, R@10 8.12, MedR 120.00
Text-to-image: R@1 1.25, R@5 6.25, R@10 8.75, MedR 108.50
Matched-minus-random similarity gap: 0.0154
```

Three-way comparison:

| Model | I->T R@5 | I->T R@10 | I->T MedR | T->I R@5 | T->I R@10 | T->I MedR | Gap |
|-------|----------|-----------|-----------|----------|-----------|-----------|-----|
| Vanilla OpenAI CLIP | 1.56 | 3.12 | 162.50 | 2.81 | 4.06 | 166.00 | 0.0005 |
| BioMedCLIP | 5.31 | 8.12 | 120.00 | 6.25 | 8.75 | 108.50 | 0.0154 |
| Retrained CLIP-norm MedCLIP | 12.81 | 20.62 | 49.50 | 11.88 | 19.38 | 47.00 | 0.1717 |

Interpretation:

- BioMedCLIP is a stronger zero-shot retrieval baseline than vanilla OpenAI CLIP.
- The project-specific fine-tuned checkpoint is still much stronger on exact OpenI image/report retrieval.
- This suggests medical-domain pretraining helps, but matching this dataset's report style and pairing distribution requires task-specific contrastive fine-tuning.

## Next Stage 3 Steps

1. Treat `configs/clipnorm.yaml` as the stronger current training setting.
2. Add image thumbnails to the qualitative markdown outputs for the CLIP-norm checkpoint.
3. Add a stronger nearest-neighbor diagnostic beyond t-SNE.
4. Consider fine-tuning from a medical pretrained model if the goal shifts from learning mechanics to maximizing retrieval quality.
