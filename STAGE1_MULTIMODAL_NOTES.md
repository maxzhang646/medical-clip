# Stage 1 Notes: CLIP-Style Multimodal Alignment

## What This Stage Is About

The goal of Stage 1 is to understand the current project as a CLIP-style multimodal alignment system.

The model is not primarily learning disease labels. It is learning this relationship:

```text
matched chest X-ray image <-> matched radiology report
```

After training, image embeddings and report embeddings should live in the same vector space. A matched image-report pair should have high cosine similarity, while mismatched pairs should have lower similarity.

## One Training Example

In `src/dataset.py`, each OpenI sample is built from:

- one frontal chest X-ray image
- one text caption made from `findings + impression`

The dataset returns:

```python
{
    "image": image_tensor,
    "input_ids": tokenized_report_ids,
    "attention_mask": tokenized_report_mask,
}
```

The important multimodal point is that the image and text are processed separately at first. They only meet later through the contrastive loss.

You can inspect a real OpenI batch with:

```bash
python3 scripts/stage1_inspect_openi_batch.py --batch-size 2 --split train
```

Expected shape pattern:

```text
image shape:           (2, 3, 224, 224)
input_ids shape:       (2, 128)
attention_mask shape:  (2, 128)
```

This means:

- `2` is the batch size
- `3` is the image channel count after converting X-rays to 3-channel tensors
- `224 x 224` is the image size expected by the visual encoder
- `128` is the maximum token length for the report text

## Stage 1 Pipeline Diagram

The full Stage 1 data flow is:

```text
OpenI X-ray image                 OpenI radiology report
        |                                  |
        v                                  v
image transforms                    ClinicalBERT tokenizer
        |                                  |
        v                                  v
image tensor                         input_ids + attention_mask
(B, 3, 224, 224)                     (B, 128)
        |                                  |
        v                                  v
CLIP ViT-B/32 visual encoder         ClinicalBERT text encoder
        |                                  |
        v                                  v
image feature                        text feature
(B, 512)                             (B, 768)
        |                                  |
        v                                  v
image projection head                text projection head
Linear(512 -> 512)                   Linear(768 -> 512)
        |                                  |
        v                                  v
L2-normalized image embedding        L2-normalized text embedding
(B, 512)                             (B, 512)
        \                                  /
         \                                /
          v                              v
             similarity matrix = image_emb @ text_emb.T
                          (B, B)
                            |
                            v
                 symmetric InfoNCE loss
```

The key transition is that two different modalities become vectors with the same shape:

```text
image embedding: (B, 512)
text embedding:  (B, 512)
```

Once both modalities are in this shared embedding space, the model can compare them with dot product / cosine similarity.

## Dual-Encoder Architecture

The model in `src/model.py` has two independent encoders:

```text
image -> OpenAI CLIP ViT-B/32 visual encoder -> 512-d visual feature
text  -> ClinicalBERT text encoder            -> 768-d text feature
```

Those two features have different origins and different dimensions, so the model uses projection heads:

```text
image feature 512 -> Linear(512, embed_dim)
text feature  768 -> Linear(768, embed_dim)
```

Both projected vectors are then L2-normalized:

```python
F.normalize(..., dim=-1)
```

This makes dot product behave like cosine similarity.

## Why Projection Heads Matter

The image encoder was pretrained for general CLIP visual features. The text encoder was pretrained as ClinicalBERT. These two models were not originally trained together.

The projection heads learn how to map both encoders into a shared embedding space. Without this shared space, image and text vectors would not be directly comparable.

## Forward Pass

For a batch of `N` image-report pairs:

```text
images -> image embeddings: shape (N, D)
texts  -> text embeddings:  shape (N, D)
```

The model computes:

```python
logits_per_image = scale * img_emb @ text_emb.T
```

This creates an `N x N` similarity matrix.

You can run a real OpenI batch through the current model with:

```bash
python3 scripts/stage1_forward_real_batch.py --batch-size 2 --split train
```

To inspect the trained model's alignment and InfoNCE loss, run:

```bash
python3 scripts/stage1_forward_real_batch.py --checkpoint checkpoints/best.pt
```

Expected shape pattern:

```text
Input image:       (2, 3, 224, 224)
Input token ids:   (2, 128)
Image embeddings:  (2, 512)
Text embeddings:   (2, 512)
Similarity matrix: (2, 2)
```

The transition from `(2, 3, 224, 224)` and `(2, 128)` to `(2, 512)` is the core multimodal representation step: two different input types become comparable vectors in one shared embedding space.

The same script also prints:

```text
labels: [0, 1]
image-to-text loss
text-to-image loss
symmetric loss
image-to-text predictions
text-to-image predictions
```

For batch size 2, `labels: [0, 1]` means `image_0` should select `report_0`, and `image_1` should select `report_1`.

Example with batch size 4:

```text
                 text_0   text_1   text_2   text_3
image_0            x       low      low      low
image_1           low       x       low      low
image_2           low      low       x       low
image_3           low      low      low       x
```

The diagonal is the matched image-report pair. Off-diagonal entries are mismatched pairs from the same batch.

## InfoNCE Loss

The loss in `src/loss.py` uses the diagonal as the correct label:

```python
labels = torch.arange(n)
```

For image-to-text:

```text
given image_i, choose the correct text_i from all texts in the batch
```

For text-to-image:

```text
given text_i, choose the correct image_i from all images in the batch
```

The final loss averages both directions:

```python
loss = (image_to_text_loss + text_to_image_loss) / 2
```

This is why the model supports both image-to-text retrieval and text-to-image retrieval.

## InfoNCE Formula

For a batch of `B` matched image-report pairs, the model creates:

```text
image embeddings: I_0, I_1, ..., I_(B-1)
text embeddings:  T_0, T_1, ..., T_(B-1)
```

The similarity score between image `i` and report `j` is:

```text
s(i, j) = scale * cosine(I_i, T_j)
```

Because the embeddings are L2-normalized, cosine similarity is computed as a dot product:

```text
cosine(I_i, T_j) = I_i dot T_j
```

For image-to-text loss, each image must classify its matching report among all reports in the batch:

```text
P(correct report j = i | image i)
    = exp(s(i, i)) / sum_j exp(s(i, j))
```

The image-to-text loss is:

```text
L_i2t = - mean_i log P(correct report j = i | image i)
```

For text-to-image loss, each report must classify its matching image among all images in the batch:

```text
P(correct image i = j | report j)
    = exp(s(j, j)) / sum_i exp(s(i, j))
```

The text-to-image loss is:

```text
L_t2i = - mean_j log P(correct image i = j | report j)
```

The final symmetric loss is:

```text
L = (L_i2t + L_t2i) / 2
```

In code, this is implemented with cross entropy:

```python
labels = torch.arange(batch_size)
loss_i = F.cross_entropy(logits_per_image, labels)
loss_t = F.cross_entropy(logits_per_text, labels)
loss = (loss_i + loss_t) / 2
```

The labels are `[0, 1, 2, ...]` because the correct pairs are on the diagonal.

## Toy Example

Suppose the batch has 3 matched pairs:

```text
(image_0, report_0)
(image_1, report_1)
(image_2, report_2)
```

The model produces this similarity matrix:

```text
             report_0  report_1  report_2
image_0        4.2       1.1       0.7
image_1        0.4       3.8       1.3
image_2        1.5       0.8       4.5
```

The correct labels are:

```python
[0, 1, 2]
```

So:

- `image_0` should pick `report_0`
- `image_1` should pick `report_1`
- `image_2` should pick `report_2`

If the diagonal scores are highest, loss is low. If an off-diagonal score is higher than the correct diagonal score, loss increases.

You can run the toy version of this idea with:

```bash
python3 scripts/stage1_toy_infonce.py
```

This script creates fake image and text embeddings, prints the similarity matrix, shows the diagonal labels, and computes the symmetric image-to-text/text-to-image loss.

## Why Batch Size Matters

With batch size `N`, each image sees:

- 1 positive text
- `N - 1` negative texts

Each text also sees:

- 1 positive image
- `N - 1` negative images

Larger batches create more in-batch negatives, which usually improves contrastive learning. This is especially important in CLIP-style training.

## What Is Being Learned

The model is not directly told:

```text
this image has pneumonia
this image has cardiomegaly
```

Instead, it learns from natural report text. If reports mention words such as cardiomegaly, effusion, opacity, or pneumothorax, the model can gradually associate those text patterns with visual patterns.

This is why zero-shot classification is possible later:

```text
image embedding ~ text embedding("A patient with pneumothorax")
```

But zero-shot classification is only a probe. The main learned skill is image-report alignment.

## How Training Works

The training loop in `src/train.py` does:

```text
1. Load a batch of paired images and tokenized reports.
2. Encode images and reports.
3. Compute image-text similarity matrix.
4. Apply symmetric InfoNCE loss.
5. Backpropagate into encoders and projection heads.
6. Save the best checkpoint by validation loss.
```

The model uses two learning rates:

- lower learning rate for pretrained encoders
- higher learning rate for projection heads

That is reasonable because projection heads start from scratch, while encoders already contain useful pretrained knowledge.

## Comparison With Standard OpenAI CLIP

Standard CLIP:

```text
image encoder: CLIP visual encoder
text encoder:  CLIP text encoder
training data: large general image-caption pairs
```

This project:

```text
image encoder: CLIP visual encoder
text encoder:  ClinicalBERT
training data: chest X-ray and radiology report pairs
```

The benefit is domain adaptation to medical language. The cost is that the two encoders were not originally pretrained together, so alignment must be learned from a small medical dataset.

## Key Questions To Be Able To Answer

By the end of Stage 1, you should be able to answer:

1. What are the two modalities in this project?
2. Why does the model use two encoders?
3. Why do image and text features need projection heads?
4. What does L2 normalization do?
5. What is inside the `N x N` similarity matrix?
6. Why is the diagonal the correct label?
7. Why does the loss run in both image-to-text and text-to-image directions?
8. Why does a larger batch size help contrastive learning?
9. Why can this model do retrieval after training?
10. Why is zero-shot classification possible but not the central claim?

## Key Questions Answer Key

1. **What are the two modalities in this project?**

   The two modalities are chest X-ray images and radiology report text.

2. **Why does the model use two encoders?**

   Images and text have different raw formats. An image is a pixel tensor, while a report is a sequence of tokens. The model needs an image encoder for visual information and a text encoder for language information.

3. **Why do image and text features need projection heads?**

   The CLIP visual encoder outputs 512-dimensional features, while ClinicalBERT outputs 768-dimensional features. Projection heads map both into the same 512-dimensional embedding space so they can be compared directly.

4. **What does L2 normalization do?**

   L2 normalization scales each embedding to unit length. After this, dot product mainly measures direction similarity, so it behaves like cosine similarity.

5. **What is inside the `N x N` similarity matrix?**

   Each row is an image, each column is a report, and each cell is the similarity score between that image and that report.

6. **Why is the diagonal the correct label?**

   The dataloader returns matched image-report pairs in the same order. Therefore `image_0` matches `report_0`, `image_1` matches `report_1`, and so on. These matched pairs sit on the diagonal.

7. **Why does the loss run in both image-to-text and text-to-image directions?**

   The model should support both retrieval directions: finding a report from an image and finding an image from a report. Symmetric loss trains both behaviors.

8. **Why does a larger batch size help contrastive learning?**

   With batch size `N`, each sample gets `N - 1` in-batch negatives. A larger batch gives the model more mismatched examples to push away from the correct pair.

9. **Why can this model do retrieval after training?**

   After training, matched images and reports should be close in the shared embedding space. Retrieval is just nearest-neighbor search by similarity.

10. **Why is zero-shot classification possible but not the central claim?**

    Zero-shot classification is possible because disease prompts can be embedded as text and compared with image embeddings. It is not the central claim because the model was trained on image-report alignment, not supervised disease labels.

## Current Implementation Notes

These are important for later stages:

- The model uses OpenAI CLIP's ViT-B/32 visual encoder.
- The model uses `medicalai/ClinicalBERT` as the text encoder.
- The first several CLIP visual transformer blocks can be frozen.
- The text feature uses the ClinicalBERT `[CLS]` token.
- The loss is symmetric image-text contrastive loss.
- The current project should be understood first as a retriever and alignment model.
