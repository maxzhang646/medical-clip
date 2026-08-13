# Medical CLIP for Chest X-ray Vision-Language Alignment and Evidence Retrieval

## Project Goals

- Learn CLIP-style multimodal alignment between chest X-ray images and radiology reports
- Train a chest X-ray image-report retriever with contrastive learning
- Evaluate bidirectional image-text retrieval (Recall@K, MedR) as the main alignment signal
- Use zero-shot classification on 8 NIH ChestX-ray14 disease classes as a secondary probe of the learned embedding space
- Study prompt design and its effect on zero-shot performance
- Explore evidence retrieval as a lightweight application of the learned multimodal embedding space

---

## Datasets

### Training: Indiana University OpenI
- **Kaggle slug**: `raddar/chest-xrays-indiana-university`
- **URL**: https://www.kaggle.com/datasets/raddar/chest-xrays-indiana-university
- **Size**: ~7,470 chest X-ray images + 3,955 free-text radiology reports
- **License**: CC BY-NC-ND 4.0
- **Why**: Only fully-public Kaggle dataset with real radiology reports (not just labels)
- **Report structure**: Each report has 4 sections — Comparison, Indication, Findings, Impression
- **Training text**: Concatenate `Findings` + `Impression` as the caption for each image

### Evaluation: NIH ChestX-ray14
- **Kaggle slug**: `nih-chest-xrays/data`
- **URL**: https://www.kaggle.com/datasets/nih-chest-xrays/data
- **Size**: 112,120 images from 30,805+ patients, 14 disease labels
- **License**: Public Domain / CC0
- **Why**: Large, clean label set for zero-shot evaluation; no reports needed at eval time

### Download Commands
```bash
kaggle datasets download raddar/chest-xrays-indiana-university
kaggle datasets download nih-chest-xrays/data
```

---

## 8 Disease Classes for Zero-shot

All 8 exist in both NIH labels and commonly appear in OpenI reports:

1. Atelectasis
2. Cardiomegaly
3. Consolidation
4. Edema
5. Effusion
6. Infiltration
7. Pneumonia
8. Pneumothorax

---

## Model Architecture

```
Image Encoder               Text Encoder
OpenAI ViT-B/32        +   ClinicalBERT (medicalai/ClinicalBERT)
(pretrained CLIP visual)    (replaces CLIP's original text tower)
        │                           │
   Linear(512→512)           Linear(768→512)
        │                           │
        └──── cosine similarity ─────┘
                  InfoNCE loss  (τ = 0.07)
```

**Design rationale**:
- Start from OpenAI CLIP image encoder weights — strong ImageNet visual features, no need to train from scratch
- Swap text encoder to ClinicalBERT — radiology reports use very different vocabulary from CLIP's LAION training data
- Both projection heads map to shared 512-d L2-normalized embedding space
- Freeze first 8 image encoder layers, fine-tune the rest

### Model Code Skeleton

```python
import clip
import torch
import torch.nn as nn
from transformers import AutoModel
import math

class MedCLIP(nn.Module):
    def __init__(self):
        super().__init__()
        clip_model, _ = clip.load("ViT-B/32")
        self.image_encoder = clip_model.visual
        self.text_encoder  = AutoModel.from_pretrained("medicalai/ClinicalBERT")
        self.image_proj    = nn.Linear(512, 512)
        self.text_proj     = nn.Linear(768, 512)
        self.logit_scale   = nn.Parameter(torch.ones([]) * math.log(1 / 0.07))

    def encode_image(self, images):
        feats = self.image_encoder(images).float()
        return nn.functional.normalize(self.image_proj(feats), dim=-1)

    def encode_text(self, input_ids, attention_mask):
        out = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        feats = out.last_hidden_state[:, 0]  # [CLS] token
        return nn.functional.normalize(self.text_proj(feats), dim=-1)

    def forward(self, images, input_ids, attention_mask):
        img_emb  = self.encode_image(images)
        text_emb = self.encode_text(input_ids, attention_mask)
        scale    = self.logit_scale.exp()
        logits   = scale * img_emb @ text_emb.T
        return logits
```

---

## Project Structure

```
medical_clip/
├── data/
│   ├── indiana/                # OpenI images + XML reports
│   └── nih/                    # NIH ChestX-ray14 images + Data_Entry_2017.csv
├── src/
│   ├── dataset.py              # OpenIDataset, NIHDataset
│   ├── model.py                # MedCLIP class
│   ├── loss.py                 # InfoNCE / NT-Xent loss
│   ├── train.py                # training loop
│   ├── retrieval.py            # Recall@K, MedR evaluation
│   ├── zeroshot.py             # zero-shot classifier
│   └── prompts.py              # prompt template library
├── notebooks/
│   ├── 01_eda.ipynb            # data exploration and visualization
│   ├── 02_train.ipynb          # training + loss curves
│   ├── 03_retrieval.ipynb      # retrieval metrics + qualitative demo
│   └── 04_zeroshot.ipynb       # zero-shot + prompt ablation
├── configs/
│   └── base.yaml               # all hyperparameters
└── requirements.txt
```

---

## Training Configuration

```yaml
# configs/base.yaml
model:
  image_encoder: ViT-B/32
  text_encoder: medicalai/ClinicalBERT
  embed_dim: 512
  freeze_image_layers: 8      # freeze first 8 transformer blocks

training:
  batch_size: 128             # larger = more negatives = better contrastive signal
  epochs: 50
  optimizer: AdamW
  lr_projections: 1e-4
  lr_encoders: 1e-5           # lower LR for pretrained encoders
  warmup_ratio: 0.1
  lr_schedule: cosine
  temperature: 0.07
  fp16: true

data:
  train_split: 0.8
  val_split: 0.1
  test_split: 0.1
  image_size: 224
  stratify_by: patient_id     # avoid patient leakage across splits
```

---

## Data Splits

- **OpenI**: 80/10/10 train/val/test, stratified by patient ID (not image) to prevent leakage
- **NIH**: Used only for zero-shot evaluation — never seen during training

---

## Augmentations

```python
train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

val_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])
```

---

## Evaluation

### Retrieval (on OpenI test split)

Metrics for both directions (image→text and text→image):

| Metric | Description |
|--------|-------------|
| Recall@1 | Correct match is the top-1 result |
| Recall@5 | Correct match is in top-5 results |
| Recall@10 | Correct match is in top-10 results |
| MedR | Median rank of correct match (lower = better) |

**Baseline**: Raw OpenAI CLIP (no fine-tuning) on same test set — shows medical fine-tuning helps.

### Zero-shot Classification (on NIH)

For each disease class, define positive and negative prompt sets, average their embeddings, classify by cosine similarity.

Metrics per class and averaged:
- AUC-ROC
- F1 score
- Accuracy

**Baseline**: Raw OpenAI CLIP zero-shot on same NIH classes.

---

## Prompt Design Ablation

This is the most differentiating part of the project. Test these prompt families:

```python
PROMPT_TEMPLATES = {
    "simple":       "{disease}",
    "findings":     "Findings of {disease} in chest X-ray",
    "clinical":     "The chest radiograph demonstrates {disease}",
    "patient":      "A patient with {disease}",
    "radiologist":  "There is evidence of {disease}",
    "normal":       "No evidence of {disease}",       # negative anchor
    "ensemble":     # average embeddings of all positive prompts
}
```

**Expected results table** (fill in with your actual numbers):

| Prompt Set | AUC Pneumonia | AUC Effusion | AUC Cardiomegaly | Macro AUC |
|------------|--------------|--------------|------------------|-----------|
| simple | | | | |
| findings | | | | |
| clinical | | | | |
| patient | | | | |
| radiologist | | | | |
| ensemble | | | | |
| positive + negative anchors | | | | |

**Key findings to look for**:
- Ensemble almost always wins — more descriptive context = better embedding
- Clinical phrasing outperforms bare label names
- Negative anchors help for imbalanced classes (Hernia, Pneumonia are rare in NIH)

---

## Visualizations

1. **EDA (Notebook 01)**
   - Report length histogram (tokens per report)
   - Disease co-occurrence heatmap from OpenI impression text
   - Sample image grid (4×4) with truncated captions

2. **Training (Notebook 02)**
   - Train vs val InfoNCE loss curves
   - t-SNE of image embeddings colored by disease class (after training)
   - t-SNE comparison: before vs after fine-tuning

3. **Retrieval (Notebook 03)**
   - Qualitative demo: query image → top-5 retrieved captions (with images)
   - Recall@K bar chart: MedCLIP vs baseline CLIP

4. **Zero-shot (Notebook 04)**
   - ROC curves per disease (8 subplots)
   - Prompt ablation bar chart (macro AUC by prompt type)

---

## Key Libraries

```
# requirements.txt
torch>=2.0
torchvision
open-clip-torch          # or pip install git+https://github.com/openai/CLIP.git
transformers>=4.30
scikit-learn
faiss-cpu                # fast approximate nearest-neighbor for retrieval at scale
matplotlib
seaborn
pandas
pillow
pyyaml
kaggle                   # CLI for dataset download
```

---

## Implementation Timeline

| Day | Task |
|-----|------|
| 1–2 | Download data, parse OpenI XMLs, build OpenIDataset and NIHDataset |
| 3 | EDA notebook — report lengths, disease distribution, sample images |
| 4–5 | Implement MedCLIP model, InfoNCE loss, training loop |
| 6–7 | Train on OpenI, track loss curves, checkpoint best model |
| 8 | Implement retrieval evaluation (Recall@K, MedR) |
| 9 | Implement zero-shot classifier, run on NIH |
| 10–11 | Prompt ablation study, generate tables and plots |
| 12 | t-SNE visualization, qualitative retrieval demo |
| 13–14 | Report writing and final cleanup |

---

## Scope Check

| Component | Difficulty | Est. Days | Impact on Grade |
|-----------|-----------|-----------|-----------------|
| Data pipeline | Low | 2 | Baseline |
| Model + training | Medium | 3–4 | Core |
| Retrieval eval | Low | 1 | High |
| Zero-shot eval | Low | 1 | High |
| Prompt ablation | Low | 2 | **Highest — most novel angle** |
| Visualizations | Low | 1 | Presentation |

Training on OpenI (~7K images) runs in under 1 hour on a Kaggle T4 GPU. No expensive compute needed.

---

## Why This Project Stands Out

The combination of:
- **Contrastive learning** (InfoNCE / CLIP paradigm)
- **Domain adaptation** (general → medical vision-language)
- **Zero-shot inference** (no labeled training data at test time)
- **Prompt engineering analysis** (connects to active LLM/VLM research)

...covers more ML subfields than a standard classifier project and produces clean, publishable-style figures. The prompt ablation table alone makes for a compelling results section.
