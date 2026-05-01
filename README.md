# Medical CLIP: Chest X-ray Retrieval and Zero-shot Diagnosis

A medical vision-language model trained on radiology report pairs via contrastive learning. Supports image-text retrieval and zero-shot disease classification on 8 pathology classes — no labeled training data required at inference time.

---

## Results (to be updated)

| Task | Metric | MedCLIP (ours) | Vanilla CLIP baseline |
|------|--------|---------------|----------------------|
| Image→Text Retrieval | Recall@1 | — | — |
| Image→Text Retrieval | Recall@5 | — | — |
| Text→Image Retrieval | Recall@1 | — | — |
| Zero-shot Classification | Macro AUC | — | — |

---

## Method

We fine-tune a CLIP-style model on (X-ray image, radiology report) pairs using InfoNCE contrastive loss. The image encoder starts from OpenAI's pretrained ViT-B/32; the text encoder uses ClinicalBERT, pretrained on medical records.

```
X-ray image   →  ViT-B/32  →  Linear(512→512)  ─┐
                                                   ├─ cosine sim → InfoNCE loss
Radiology report → ClinicalBERT → Linear(768→512) ─┘
```

At inference, zero-shot classification works by comparing image embeddings against text prompt embeddings — no task-specific fine-tuning needed.

---

## Datasets

| Dataset | Role | Size | Source |
|---------|------|------|--------|
| Indiana University OpenI | Training (image + report pairs) | 7,470 images, 3,955 reports | [Kaggle](https://www.kaggle.com/datasets/raddar/chest-xrays-indiana-university) |
| NIH ChestX-ray14 | Zero-shot evaluation | 112,120 images, 14 disease labels | [Kaggle](https://www.kaggle.com/datasets/nih-chest-xrays/data) |

### Download

```bash
# Requires Kaggle API credentials (~/.kaggle/kaggle.json)
kaggle datasets download raddar/chest-xrays-indiana-university -p data/indiana --unzip
kaggle datasets download nih-chest-xrays/data -p data/nih --unzip
```

---

## Zero-shot Disease Classes (NIH)

Atelectasis · Cardiomegaly · Consolidation · Edema · Effusion · Infiltration · Pneumonia · Pneumothorax

---

## Prompt Engineering Ablation

A key analysis in this project is how prompt design affects zero-shot accuracy:

| Prompt Template | Example |
|----------------|---------|
| Simple | `"Pneumonia"` |
| Findings | `"Findings of pneumonia in chest X-ray"` |
| Clinical | `"The chest radiograph demonstrates pneumonia"` |
| Patient | `"A patient with pneumonia"` |
| Ensemble | Average of all positive prompt embeddings |
| Pos + Neg anchors | Ensemble positive minus `"No evidence of pneumonia"` |

---

## Project Structure

```
medical_clip/
├── data/
│   ├── indiana/          # OpenI images + XML reports
│   └── nih/              # NIH images + Data_Entry_2017.csv
├── src/
│   ├── dataset.py        # OpenIDataset, NIHDataset
│   ├── model.py          # MedCLIP model class
│   ├── loss.py           # InfoNCE loss
│   ├── train.py          # training loop
│   ├── retrieval.py      # Recall@K, MedR evaluation
│   ├── zeroshot.py       # zero-shot classifier
│   └── prompts.py        # prompt template library
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_train.ipynb
│   ├── 03_retrieval.ipynb
│   └── 04_zeroshot.ipynb
├── configs/
│   └── base.yaml
├── scripts/
│   ├── download_data.sh
│   └── train.sh
├── requirements.txt
└── README.md
```

---

## Setup

```bash
git clone https://github.com/maxzhang646/medical-clip.git
cd medical-clip
pip install -r requirements.txt
```

### Requirements
- Python 3.10+
- PyTorch 2.0+ (MPS supported for Apple Silicon)
- See `requirements.txt` for full list

---

## Training

```bash
python src/train.py --config configs/base.yaml
```

The training script auto-detects MPS (Apple Silicon) / CUDA / CPU.

---

## Evaluation

```bash
# Retrieval on OpenI test split
python src/retrieval.py --checkpoint checkpoints/best.pt

# Zero-shot classification on NIH
python src/zeroshot.py --checkpoint checkpoints/best.pt --prompt ensemble
```

---

## Roadmap

- [x] Project plan and repo setup
- [ ] Data pipeline (OpenIDataset, NIHDataset)
- [ ] MedCLIP model implementation
- [ ] Training loop with MPS support
- [ ] Retrieval evaluation (Recall@K, MedR)
- [ ] Zero-shot classification on NIH 8 classes
- [ ] Prompt engineering ablation study
- [ ] LLM synthetic report augmentation experiment

---

## References

- [OpenAI CLIP](https://arxiv.org/abs/2103.00020) — Radford et al., 2021
- [ConVIRT](https://arxiv.org/abs/2010.00747) — Zhang et al., 2020
- [CheXzero](https://www.nature.com/articles/s41551-022-00936-9) — Tiu et al., 2022
- [MedCLIP](https://arxiv.org/abs/2210.10163) — Wang et al., 2022
- [ClinicalBERT](https://arxiv.org/abs/1904.05342) — Alsentzer et al., 2019
