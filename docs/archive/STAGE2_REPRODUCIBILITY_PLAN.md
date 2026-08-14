# Stage 2 Plan: Reproducibility Aligned With Stage 1

## Goal

Stage 1 proved that the multimodal pipeline works conceptually:

```text
OpenI image/report
-> image tensor + token ids
-> image/text embeddings
-> similarity matrix
-> symmetric InfoNCE loss
```

Stage 2 should make that same pipeline reproducible from a fresh checkout. The goal is not to improve model quality yet. The goal is to make every Stage 1 step rerunnable with clear paths, dependencies, configs, and deterministic data splits.

## Stage 2 Success Criteria

By the end of Stage 2, a fresh user should be able to run:

```bash
python3 scripts/stage1_toy_infonce.py
python3 scripts/stage1_inspect_openi_batch.py --batch-size 2 --split train
python3 scripts/stage1_forward_real_batch.py --batch-size 2 --split train
python3 scripts/stage1_forward_real_batch.py --checkpoint checkpoints/best.pt
```

without manually editing source files or guessing dataset paths.

## Step 1: Make Data Paths Portable

Current issue:

```yaml
data:
  indiana_dir: /Users/wenluzhang/.cache/kagglehub/datasets/raddar/chest-xrays-indiana-university/versions/2
```

This is machine-specific. It worked in Stage 1 because the data exists on this machine, but it will break elsewhere.

Tasks:
- Replace the absolute OpenI path in `configs/base.yaml` with a repo-relative default such as `data/indiana`.
- Allow command-line scripts to override `indiana_dir` if the dataset lives elsewhere.
- Document the exact expected OpenI structure:

```text
data/indiana/
  indiana_reports.csv
  indiana_projections.csv
  images/
    images_normalized/
      *.png
```

Why this aligns with Stage 1:
- `scripts/stage1_inspect_openi_batch.py` and `scripts/stage1_forward_real_batch.py` both depend on `OpenIDataset`.
- If `OpenIDataset` cannot find the same files, the Stage 1 pipeline cannot be reproduced.

Deliverable:
- Portable dataset config and README setup instructions.

## Step 2: Add Explicit Dependency Coverage

Current issue:
- `src/model.py` imports `clip`, but `requirements.txt` does not install OpenAI CLIP.
- Stage 1 forward scripts also require `transformers`, `torch`, `torchvision`, `Pillow`, `pandas`, and `pyyaml`.

Tasks:
- Add the correct OpenAI CLIP installation requirement.
- Separate core runtime dependencies from optional notebook/demo dependencies if useful.
- Add a short dependency check script or command.

Minimum expected dependency set for Stage 1:

```text
torch
torchvision
transformers
Pillow
pandas
pyyaml
ftfy
regex
tqdm
openai-clip or git+https://github.com/openai/CLIP.git
```

Why this aligns with Stage 1:
- `stage1_toy_infonce.py` only needs PyTorch.
- `stage1_inspect_openi_batch.py` needs dataset/tokenizer dependencies.
- `stage1_forward_real_batch.py` needs the full model stack, including CLIP and ClinicalBERT.

Deliverable:
- Updated `requirements.txt` that can install everything needed for Stage 1 scripts.

## Step 3: Make Splits Deterministic and Inspectable

Current behavior:
- `OpenIDataset._load_samples` shuffles `uid`s with `random_state=42` and splits inside the dataset class.
- This is deterministic, but the split is implicit and not saved.

Tasks:
- Create a split generation script, for example `scripts/create_openi_splits.py`.
- Save split files such as:

```text
splits/openi_train_uids.txt
splits/openi_val_uids.txt
splits/openi_test_uids.txt
```

- Update `OpenIDataset` to optionally load split files.
- Keep the existing deterministic fallback for convenience.

Why this aligns with Stage 1:
- Stage 1 examples use `split=train`.
- If the train split changes silently, the examples and reported outputs become hard to compare.

Deliverable:
- Fixed split files or a deterministic split generation script.

## Step 4: Make Config Values Actually Drive the Code

Current issue:
- Some config values exist but are not consistently used.
- `train_split`, `val_split`, and `test_split` are listed in config, but `OpenIDataset` currently hardcodes 80/10/10.
- `temperature` is listed in config, but `MedCLIP` initializes `logit_scale` internally with `0.07`.
- `warmup_ratio` is used to compute `warmup_steps`, but warmup is not actually applied.

Tasks:
- Decide which config values are real controls and which should be removed.
- Pass split ratios into split generation or document that splits are fixed.
- Pass temperature into `MedCLIP` or document it as fixed.
- Either implement warmup or remove `warmup_ratio` from config.

Why this aligns with Stage 1:
- Stage 1 teaches that `temperature/logit_scale` affects the similarity logits used by InfoNCE.
- If config says one thing and code does another, the learning artifact becomes misleading.

Deliverable:
- Config file that accurately reflects real behavior.

## Step 5: Add Stage 1 Smoke Tests

Goal:
- Convert the Stage 1 learning scripts into reproducibility checks.

Tasks:
- Add a short smoke test command section to README.
- Optionally add `scripts/stage2_smoke_check.py` that runs:

```text
1. import key modules
2. load config
3. verify OpenI files exist
4. instantiate OpenIDataset
5. load one batch
6. optionally run model forward if model weights are available
```

Why this aligns with Stage 1:
- The smoke check should validate the same path documented in `STAGE1_MULTIMODAL_NOTES.md`.

Deliverable:
- A command that quickly answers: "Can this repo run the Stage 1 multimodal pipeline on this machine?"

## Step 6: Clarify Checkpoint Expectations

Current issue:
- Stage 1 can run `--checkpoint checkpoints/best.pt` on this machine, but a fresh clone may not include or download that checkpoint.

Tasks:
- Document whether `checkpoints/best.pt` is committed, generated, or manually supplied.
- If not committed, document how to train it or where to place it.
- Make scripts print a clear error if checkpoint path is missing.

Why this aligns with Stage 1:
- The trained checkpoint is what makes the diagonal alignment example convincing.
- Without clear checkpoint handling, users may confuse random initialization with trained alignment.

Deliverable:
- README checkpoint section and friendlier script error handling.

## Step 7: Update README With a Clean Reproduction Path

The README should include a minimal reproduction path:

```bash
pip install -r requirements.txt
bash scripts/download_data.sh
python3 scripts/stage1_toy_infonce.py
python3 scripts/stage1_inspect_openi_batch.py --batch-size 2 --split train
python3 scripts/stage1_forward_real_batch.py --batch-size 2 --split train
python3 scripts/stage1_forward_real_batch.py --checkpoint checkpoints/best.pt
```

If `bash scripts/download_data.sh` downloads into `data/indiana`, the config should match that.

Why this aligns with Stage 1:
- Stage 1 is the first thing a learner should be able to reproduce.
- README should let them rerun the exact same conceptual path.

Deliverable:
- README setup and smoke-test section that matches actual code.

## Recommended Implementation Order

1. Fix OpenI path handling.
2. Fix dependency declaration for CLIP.
3. Add split generation or split file support.
4. Clean up config/code mismatches.
5. Add Stage 1 smoke test command.
6. Improve checkpoint error messages.
7. Update README.

This order starts from the blockers that would prevent Stage 1 scripts from running, then moves toward training and evaluation reproducibility.

## Definition of Done

Stage 2 is done when:

- No required path in config points to a user-specific absolute directory.
- `requirements.txt` includes all packages needed by Stage 1 scripts and core training/evaluation code.
- OpenI split behavior is deterministic and documented.
- Config values match actual code behavior.
- Stage 1 scripts run from documented commands.
- README has a clean setup and smoke-test path.
- Checkpoint expectations are explicit.

