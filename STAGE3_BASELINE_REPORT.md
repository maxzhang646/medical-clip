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

> **这个结论在 Stage 4 被推翻了（见第 11 节）。** 这里的比较不公平：只有本项目的 checkpoint 在
> OpenI 上训练过，BioMedCLIP 是 zero-shot。把 BioMedCLIP 用同样的数据和预算微调之后，它在所有
> ranking 指标上都反超。

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
4. 当前项目 checkpoint 在 OpenI exact retrieval 上优于 **off-the-shelf（未微调的）** BioMedCLIP baseline。
5. Stage 4 补上了缺失的对照：微调后的 BioMedCLIP 反超本项目 checkpoint（见第 11 节）。

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
- Stage 3 的 BioMedCLIP baseline 是 zero-shot/off-the-shelf（Stage 4 已补上 fine-tuned 版本）
- Stage 4 的对比仍有残留混淆：ViT-B/16 vs B/32（见 11.6）

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

## 11. Stage 4: 微调后的 BioMedCLIP（补全 2x2）

### 11.1 为什么要做

第 4 节那张表混淆了两个变量：初始化来源、是否在 OpenI 上微调。摊成 2x2 会发现缺一格：

```text
                        zero-shot        fine-tuned on OpenI
CLIP + ClinicalBERT       done                 done
BioMedCLIP                done                 缺
```

缺的这格是唯一能拆开两个变量的。不补上，"我们比 BioMedCLIP 好"就只是"只有我们训练过"。

### 11.2 实验设置

除 `lr_encoders` 外，全部与 `configs/clipnorm.yaml` 一致：同一份 splits、batch 64、15 epochs、
freeze 8 个 image block、cosine + warmup。三个 LR arm 在 Kaggle T4 上各跑约 81 分钟。

关键实现约束：**没有外接任何新的 projection layer**。BioMedCLIP 自带训练好的投影头，加一个随机初始化
的 Linear 会当场摧毁预训练对齐，实验就失去意义。同理保留它自己的 `logit_scale`（约 0.0117），而不是
config 里的 0.07。

### 11.3 结果

| Model | I->T R@1 | R@5 | R@10 | MedR | T->I R@5 | Gap |
|-------|----------|-----|------|------|----------|-----|
| Vanilla OpenAI CLIP (zero-shot) | 0.00 | 1.56 | 3.12 | 162.50 | 2.81 | 0.0005 |
| BioMedCLIP (zero-shot) | 1.56 | 5.31 | 8.12 | 120.00 | 6.25 | 0.0154 |
| CLIP+ClinicalBERT (CLIP-norm, FT) | 3.75 | 12.81 | 20.62 | 49.50 | 11.88 | 0.1717 |
| **BioMedCLIP FT (lr 1e-5)** | **6.25** | **17.50** | **27.19** | **45.50** | **17.50** | 0.0230 |
| BioMedCLIP FT (lr 3e-6) | 5.00 | 17.19 | 25.62 | 46.50 | 14.69 | 0.0232 |
| BioMedCLIP FT (lr 1e-6) | 3.12 | 12.50 | 20.00 | 67.00 | 13.44 | 0.0146 |

补全后的 2x2（I->T R@5）：

|  | zero-shot | fine-tuned |
|--|-----------|------------|
| CLIP + ClinicalBERT | 1.56 | 12.81 |
| BioMedCLIP | 5.31 | **17.50** |

两个主效应都是正的：微调在两行都有效，医学预训练在两列都有效，叠加最优。

### 11.4 关于 LR sweep

跑三个 arm 是为了让"结果不好"能归因到 LR 而不是 backbone。实际方向相反：

- 最大的 LR（1e-5）赢了，**预期中的灾难性遗忘没有出现** —— 前 3 个 epoch val_loss 从 4.53 降到 3.6278
- 最小的 LR（1e-6）明显欠拟合

但过拟合来得极快：best epoch 是 3（对照组是 9），到 epoch 15 时 val_loss 4.2297 已高于
ln(64) = 4.159 的随机基线。起点已对齐的模型，在 2554 个样本上很快就没东西可学了。

### 11.5 `gap` 指标的适用范围（重要）

BioMedCLIP FT 的检索指标全面领先，gap 却只有对照组的 1/7。这不是矛盾，是 gap 本身不是尺度无关量：

| Model | matched | random | gap | gap / random std |
|-------|---------|--------|-----|------------------|
| CLIP+ClinicalBERT FT | 0.2804 | 0.1087 | 0.1717 | 1.0965 |
| BioMedCLIP FT (lr 1e-5) | 0.4173 | 0.3942 | 0.0230 | 待补测 |

BioMedCLIP 的嵌入挤在一个很窄的锥形区域里：两个毫不相关的图文向量，余弦相似度已经有 0.394。
绝对差值被压缩，但**排序完全不受影响**。

结论：

```text
raw gap 只能在同一个 backbone 内部比较。
跨 backbone 比较必须用 R@K / MedR，或 normalized gap (gap / random std)。
```

`scripts/stage3_medclip_diagnostic.py` 现已同时输出 normalized gap，并支持
`--backbone biomedclip` 直接评测 Stage 4 的 checkpoint。

第 5.1 节用 gap 判定"vanilla CLIP 几乎没有医学对齐"仍然成立 —— 那是在同属窄锥几何的两个模型之间比较
（0.0005 vs 0.0154，46 倍），而且结论与 ranking 指标一致。

### 11.6 残留混淆

这个胜负不能完全归因于"医学预训练"：

- **图像塔不同**：BioMedCLIP 是 ViT-B/16，本项目是 ViT-B/32，一部分增益来自更细的 patch。
  BioMedCLIP 没有发布 B/32 版本，这一项无法彻底控住，必须在结论里声明。
- **context length**：256 vs 128。影响很小（典型 OpenI caption 约 53 token，两边都很少截断）。
- **temperature**：0.0117 vs 0.07，属于"预训练模型自带超参"，无法在保留对齐的前提下对齐。

### 11.7 修正后的结论

```text
在相同数据和相同微调预算下，
医学域预训练的起点显著优于"通用视觉塔 + 临床文本塔"的组装方案。
起点比微调策略更重要。
```

若目标是最大化检索性能，下一步应以 `checkpoints_biomedclip_lr1e5/best.pt` 为新基线，
并把 epochs 从 15 降到 5 左右（best epoch 是 3，其余 12 个 epoch 纯属浪费）。
