# Multimodal Learning Plan

## Project Positioning

This project is primarily a multimodal learning project, not a clinical diagnosis system.

The core learning objective is to understand how a CLIP-style model aligns two modalities:

```text
Chest X-ray image -> image encoder -> image embedding
Radiology report  -> text encoder  -> text embedding

matched image embedding ~= matched report embedding
```

Retrieval and zero-shot classification are evaluation tools for this alignment. RAG-style evidence retrieval is a later application layer, not the main focus.

## Stage 1: Understand the Current CLIP-Style Model

Goal: Be able to explain every part of the current multimodal training pipeline.

Tasks:
- Trace one training example from image file and report text to model inputs.
- Understand the image encoder, text encoder, projection heads, normalization, and similarity matrix.
- Derive the InfoNCE loss used for image-to-text and text-to-image alignment.
- Explain why larger batch sizes provide more in-batch negatives.
- Compare this architecture against standard OpenAI CLIP.

Deliverables:
- Short notes explaining `src/model.py`, `src/loss.py`, and `src/train.py`.
- A diagram of the dual-encoder pipeline.
- A toy example showing how the similarity matrix and contrastive labels work.

## Stage 2: Make the Project Reproducible

Goal: Make the project easy to rerun and trustworthy as a learning artifact.

Tasks:
- Remove machine-specific absolute paths from `configs/base.yaml`.
- Add the correct OpenAI CLIP dependency to `requirements.txt`.
- Save fixed train/val/test split files instead of regenerating splits implicitly.
- Document exact dataset locations and expected file structure.
- Make config values such as image size, batch size, and evaluation sample size consistently respected by code.

Deliverables:
- A clean setup path from fresh clone to training/evaluation.
- Fixed split files or a deterministic split generation script.
- README instructions that match the actual commands.

## Stage 3: Strengthen Multimodal Evaluation

Goal: Evaluate whether the image and text embedding spaces are actually aligned.

Tasks:
- Keep bidirectional retrieval as the primary metric: image-to-text and text-to-image Recall@K, MedR.
- Add qualitative retrieval examples with query image, ground-truth report, top retrieved reports, and failure notes.
- Add embedding visualizations with UMAP or t-SNE, colored by disease terms or report keywords.
- Compare at least three retrievers:
  - vanilla OpenAI CLIP
  - a medical pretrained model such as BioMedCLIP if feasible
  - this fine-tuned model
- Check whether retrieved reports are medically similar even when they are not the exact paired report.

Deliverables:
- `results.md` section for qualitative retrieval cases.
- A figure showing embedding clusters or retrieval examples.
- A clear baseline comparison table.

## Stage 4: Study Prompt-Based Zero-Shot Behavior

Goal: Use zero-shot classification as a probe of the learned embedding space, not as the central claim.

Tasks:
- Keep prompt ablation across simple, clinical, patient, radiologist, ensemble, and positive-negative prompts.
- Report per-class AUC and macro AUC.
- Add class prevalence for the NIH evaluation subset.
- Analyze why prompts help some diseases more than others.
- Avoid overstating zero-shot diagnosis performance.

Deliverables:
- Prompt ablation table.
- Per-disease failure analysis.
- README language that describes zero-shot classification as an evaluation probe.

## Stage 5: Improve the Retriever

Goal: Make the multimodal alignment stronger before adding more complex downstream systems.

Tasks:
- Verify image preprocessing matches the chosen visual encoder.
- Try freezing/unfreezing different numbers of image encoder layers.
- Tune batch size, learning rates, and projection head settings.
- Explore hard negatives if feasible.
- Consider using more report text variants: findings only, impression only, findings plus impression.
- Evaluate whether duplicate reports, frontal/lateral views, or patient-level grouping affect retrieval.

Deliverables:
- Ablation table for preprocessing, freezing strategy, and report text choice.
- Updated best checkpoint with documented config.

## Stage 6: Build a Small Evidence Retrieval Demo

Goal: Show a practical use of multimodal retrieval while keeping the project focused on alignment.

Tasks:
- Precompute report embeddings for the OpenI report index.
- Given a query X-ray, retrieve top-k similar reports.
- Add adaptive-k retrieval based on similarity score drops.
- Display retrieved findings and impressions as evidence.
- Optionally summarize common observations from retrieved reports.

Deliverables:
- `src/index.py` or notebook code to build the embedding index.
- `src/search.py` or notebook code to retrieve similar reports.
- A small demo with query image and top retrieved evidence.

## Stage 7: Optional RAG-Lite Extension

Goal: Connect multimodal retrieval to medical RAG ideas without turning the project into a large RAG system.

Tasks:
- Use retrieved reports as grounded context for an LLM-generated explanation.
- Add a small text knowledge index for disease descriptions or guidelines.
- Keep report retrieval image-driven and knowledge retrieval text-driven.
- Clearly label generated output as educational and non-clinical.

Deliverables:
- A minimal RAG-lite notebook or script.
- Side-by-side output with and without retrieved evidence.
- Short discussion of how retrieval affects factuality.

## Suggested Portfolio Framing

Use this framing:

> Built a CLIP-style medical vision-language model to learn alignment between chest X-rays and radiology reports via contrastive learning. Evaluated cross-modal retrieval, prompt-based zero-shot classification, prompt sensitivity, and retrieval-grounded medical evidence exploration.

Avoid framing it as:

> Built a zero-shot diagnostic model for clinical use.

The stronger and more accurate story is that this project teaches multimodal alignment, retrieval, and evidence grounding in a medical imaging setting.
