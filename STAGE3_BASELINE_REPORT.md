# Stage 3 Baseline Report: Medical Image-Text Alignment

## 1. 当前问题

这个项目的核心目标不是直接做临床诊断，而是学习：

```text
Chest X-ray image <-> Radiology report
```

两种 modality 如何被编码到同一个 shared embedding space。

目前项目使用的是 CLIP-style dual-tower 结构：

```text
Image tower: OpenAI CLIP ViT-B/32 visual encoder
Text tower: ClinicalBERT
Loss: bidirectional contrastive loss / InfoNCE
```

之前的核心疑问是：

```text
图像塔和文字塔来自不同预训练模型，会不会导致它们不 aligned？
```

Stage 3 的实验就是围绕这个问题展开：不仅看模型能不能跑，还要检查 embedding space 是否真的学到了 image/report alignment。

## 2. 实验设置

所有 baseline 都在同一个 OpenI test split 上评估：

```text
Test samples: 320 image/report pairs
```

主要指标：

- `R@1`: 正确配对 report/image 是否排在第 1 名
- `R@5`: 正确配对是否进入前 5
- `R@10`: 正确配对是否进入前 10
- `MedR`: 正确配对的 median rank，越低越好
- `matched-random gap`: matched image/report similarity 减去 random mismatched similarity

其中 `matched-random gap` 很重要，因为它直接回答：

```text
真实配对的 image/report 是否比随机错误配对更接近？
```

## 3. 对比模型

### A. Vanilla OpenAI CLIP

结构：

```text
OpenAI CLIP image encoder + OpenAI CLIP text encoder
```

这个 baseline 的意义是：两个 tower 原本就是一起预训练的，所以它们在 general-domain image/text space 中是 aligned 的。

但它没有医学领域训练，也没有 OpenI report-style 适配。

### B. Project Fine-Tuned MedCLIP

结构：

```text
OpenAI CLIP ViT-B/32 image tower + ClinicalBERT text tower
```

这是当前项目自己的模型。虽然两个 tower 来自不同来源，但通过 OpenI image/report pairs 做 contrastive fine-tuning。

### C. BioMedCLIP

结构：

```text
medical pretrained image-text dual tower
```

使用模型：

```text
hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224
```

这个 baseline 的意义是：它不是 general-domain CLIP，而是已经经过医学图文预训练的 off-the-shelf medical CLIP model。

## 4. 主要结果

| Model | I->T R@1 | I->T R@5 | I->T R@10 | I->T MedR | T->I R@1 | T->I R@5 | T->I R@10 | T->I MedR | Gap |
|-------|----------|----------|-----------|-----------|----------|----------|-----------|-----------|-----|
| Vanilla OpenAI CLIP | 0.00 | 1.56 | 3.12 | 162.50 | 0.31 | 2.81 | 4.06 | 166.00 | 0.0005 |
| BioMedCLIP | 1.56 | 5.31 | 8.12 | 120.00 | 1.25 | 6.25 | 8.75 | 108.50 | 0.0154 |
| Original ImageNet-norm MedCLIP | 3.44 | 12.19 | 18.75 | 54.00 | 3.75 | 11.88 | 18.44 | 52.50 | 0.1526 |
| Retrained CLIP-norm MedCLIP | 3.75 | 12.81 | 20.62 | 49.50 | 4.38 | 11.88 | 19.38 | 47.00 | 0.1717 |

## 5. 结果解释

### 5.1 Vanilla CLIP 说明了什么

Vanilla OpenAI CLIP 的结果很弱：

```text
I->T R@5: 1.56
T->I R@5: 2.81
matched-random gap: 0.0005
```

这说明它虽然有 general image/text alignment，但几乎不能区分 OpenI 中真实配对的 X-ray/report 和随机错误配对。

所以：

```text
general-domain alignment != medical X-ray/report alignment
```

### 5.2 BioMedCLIP 说明了什么

BioMedCLIP 明显强于 vanilla CLIP：

```text
I->T R@5: 5.31
T->I R@5: 6.25
matched-random gap: 0.0154
```

这说明医学领域预训练确实有帮助。

但 BioMedCLIP 仍然明显弱于当前项目 fine-tuned checkpoint。原因可能是：

- BioMedCLIP 没有专门适配 OpenI 的 report style
- OpenI exact image/report retrieval 要求匹配具体 paired report
- 医学预训练提供 general medical alignment，但不一定记住这个数据集的配对分布

所以：

```text
medical-domain pretraining helps,
but task-specific contrastive fine-tuning still matters.
```

### 5.3 当前项目 fine-tuning 是否有效

有效。

原始 ImageNet-normalization checkpoint：

```text
matched mean: 0.3037
random mean:  0.1510
gap:          0.1526
```

Retrained CLIP-normalization checkpoint：

```text
matched mean: 0.2804
random mean:  0.1087
gap:          0.1717
```

这说明项目自己的 contrastive fine-tuning 学到了明确的 domain-specific alignment。它主要不是把 matched pair 的绝对 similarity 拉得特别高，而是把 random mismatched pairs 明显推低。

这是一个合理的 CLIP-style 学习信号。

## 6. 关于不同来源 tower 的判断

一开始的担心是：

```text
image tower 来自 OpenAI CLIP
text tower 来自 ClinicalBERT
所以它们可能不 aligned
```

这个担心是对的，但实验结果更精确：

```text
不同来源的 tower 初始时不 aligned，
但可以通过 contrastive fine-tuning 学到 alignment。
```

当前模型的问题不是“完全没有 alignment”，而是：

```text
alignment 已经有了，但 ranking strength 还不够强。
```

也就是说，模型已经能把 random mismatched report 推远，但还不能稳定把 exact matched report 排到 top 1 或 top 5。

## 7. Preprocessing Ablation

因为 image tower 是 OpenAI CLIP ViT-B/32，我们测试了两种 normalization：

```text
ImageNet normalization
CLIP normalization
```

只在 inference 时替换 normalization，提升不明显：

| Normalization | I->T R@5 | T->I R@5 | Gap |
|---------------|----------|----------|-----|
| ImageNet | 12.19 | 11.88 | 0.1526 |
| CLIP | 12.81 | 10.94 | 0.1525 |

但用 CLIP normalization 重新训练后，结果有中等提升：

| Model | I->T R@10 | I->T MedR | T->I R@10 | T->I MedR | Gap |
|-------|-----------|-----------|-----------|-----------|-----|
| Original ImageNet-norm MedCLIP | 18.75 | 54.00 | 18.44 | 52.50 | 0.1526 |
| Retrained CLIP-norm MedCLIP | 20.62 | 49.50 | 19.38 | 47.00 | 0.1717 |

结论：

```text
CLIP normalization 值得保留为更强训练设置，
但 preprocessing 不是唯一瓶颈。
```

## 8. 当前项目状态

目前项目已经有一个比较完整的 multimodal learning story：

1. Stage 1 理解了 CLIP-style dual tower、similarity matrix、InfoNCE loss。
2. Stage 2 提高了 reproducibility，包括 fixed splits、config、README、smoke check。
3. Stage 3 已经完成多个 baseline：
   - Vanilla OpenAI CLIP
   - BioMedCLIP
   - Original project MedCLIP
   - Retrained CLIP-normalization MedCLIP
4. 当前项目 checkpoint 在 OpenI exact retrieval 上优于 off-the-shelf BioMedCLIP baseline。

这让项目叙事比之前强很多：

```text
This project does not merely run a CLIP-like model.
It evaluates whether medical image-text alignment is actually learned,
and compares general-domain, medical-pretrained, and task-fine-tuned dual encoders.
```

## 9. 现实限制

当前结果仍然不应被过度解释。

主要限制：

- OpenI training set 只有约 2554 个 paired samples
- report 文本高度重复，尤其是 normal/no acute 报告
- exact report retrieval 比 semantic retrieval 更难
- 当前模型没有使用 cross-attention
- 当前模型没有 hard negative mining
- 当前模型没有利用更大医学数据集，例如 MIMIC-CXR
- BioMedCLIP baseline 是 zero-shot/off-the-shelf，没有针对 OpenI fine-tune

所以当前项目适合定位为：

```text
medical multimodal alignment learning project
```

而不是：

```text
strong clinical diagnostic model
```

## 10. 下一步建议

优先级最高的下一步：

1. 用 `checkpoints_clipnorm/best.pt` 重新生成 qualitative retrieval examples。
2. 给 qualitative retrieval markdown 加 image thumbnails，方便人工检查 retrieved image/report 是否医学相似。
3. 做 report text ablation：
   - findings only
   - impression only
   - findings + impression
4. 做 freeze strategy ablation：
   - freeze more image layers
   - current setting
   - unfreeze more layers
5. 如果目标从学习转向性能，可以尝试从 BioMedCLIP 初始化后再 fine-tune。

目前最实际的下一步是：

```text
用 CLIP-norm checkpoint 生成 qualitative retrieval examples，并人工分析成功/失败案例。
```

因为现在我们已经知道数字变好了，但还需要看模型到底检索到了什么。
