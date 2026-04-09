# EVA SAE Reproducibility Bundle

This bundle reproduces two SAE training modes used in EVA interpretability analysis:

- `Batch-TopK SAE`
- `sae_L1_penalty`

All scripts are under `notebooks/interpretability_analysis/sae_repro_release/`.

## 1) Environment

### Option A: Local Python

```bash
cd /path/to/EVA1
pip install -r notebooks/interpretability_analysis/sae_repro_release/requirements-sae-training.txt
```

### Option B: Docker (recommended)

Build from repo root:

```bash
cd /path/to/EVA1
docker build \
  -f notebooks/interpretability_analysis/sae_repro_release/Dockerfile \
  -t eva-sae:latest \
  .
```

Run container:

```bash
docker run --gpus all --rm -it \
  -v /path/to/EVA1:/workspace/EVA1 \
  -w /workspace/EVA1 \
  eva-sae:latest
```

## 2) Download model + dataset with pinned revisions

Use fixed commit hashes from Hugging Face to avoid drift.

```bash
export HF_MODEL_REVISION=<MODEL_COMMIT_SHA>
export HF_DATA_REVISION=<DATASET_COMMIT_SHA>

huggingface-cli download GENTEL-Lab/EVA \
  --revision "$HF_MODEL_REVISION" \
  --local-dir /path/to/hf_models/EVA

huggingface-cli download GENTEL-Lab/OpenRNA-v1-114M \
  --repo-type dataset \
  --revision "$HF_DATA_REVISION" \
  --local-dir /path/to/hf_data/OpenRNA-v1-114M
```

## 3) Export runtime paths

```bash
export EVA_REPO_ROOT=/path/to/EVA1
export HF_MODEL_ROOT=/path/to/hf_models/EVA
export HF_DATA_ROOT=/path/to/hf_data/OpenRNA-v1-114M
```

Defaults if unset:

- `HF_MODEL_ROOT=$EVA_REPO_ROOT/checkpoint`
- `HF_DATA_ROOT=$EVA_REPO_ROOT/data/openrna/OpenRNA-v1-114M`

Notes:

- `Batch-TopK SAE` checkpoint default: `${HF_MODEL_ROOT}/EVA_1.4B_CLM`
- `sae_L1_penalty` checkpoint default: `${HF_MODEL_ROOT}/EVA_145M`
- FASTA is auto-detected as the largest `.fa/.fasta` file under `HF_DATA_ROOT`
- Optional explicit override: `export HF_DATA_FASTA=/path/to/train.fa`
- Optional checkpoint override: `export SAE_CKPT_DIR=/path/to/checkpoint_dir`

## 4) Validate environment and paths

```bash
python notebooks/interpretability_analysis/sae_repro_release/scripts/validate_env.py

bash notebooks/interpretability_analysis/sae_repro_release/scripts/check_hf_paths.sh batch_topk
bash notebooks/interpretability_analysis/sae_repro_release/scripts/check_hf_paths.sh sae_l1_penalty
```

## 5) Smoke test (fast sanity check)

```bash
bash notebooks/interpretability_analysis/sae_repro_release/scripts/run_all_smoke.sh
```

## 6) Full training

```bash
bash notebooks/interpretability_analysis/sae_repro_release/scripts/run_batch_topk_full.sh
bash notebooks/interpretability_analysis/sae_repro_release/scripts/run_sae_l1_penalty_full.sh

# or run both sequentially
bash notebooks/interpretability_analysis/sae_repro_release/scripts/run_all_full.sh
```

## Output locations

- Checkpoints: `notebooks/interpretability_analysis/sae_repro_release/outputs/*/checkpoints`
- Logs: `notebooks/interpretability_analysis/sae_repro_release/logs`

## Reproducibility controls

- Global random seed is set for `random` / `numpy` / `torch`.
- Deterministic flags are enabled by default in configs (`deterministic: true`).
- Full configs and smoke configs are both versioned under `configs/`.

## Validation status

- Smoke validation was executed on April 9, 2026 in GPU Docker runtime (`eva:latest`).
- Both modes (`Batch-TopK SAE`, `sae_L1_penalty`) completed smoke training and produced checkpoints.

## Common errors

- `HF_DATA_FASTA not found`: wrong `HF_DATA_ROOT` or missing FASTA files.
- `Invalid SAE_CKPT_DIR`: checkpoint directory missing `config.json` or `model_weights.pt`.
- `Import failed: megablocks...`: install dependencies again or use Docker image.
