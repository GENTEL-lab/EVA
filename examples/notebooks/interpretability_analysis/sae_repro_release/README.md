# EVA SAE Reproducibility Bundle

This bundle reproduces two SAE training modes used in EVA interpretability analysis:

- `Batch-TopK SAE`
- `sae_L1_penalty`

Engineering smoke tests do not establish the provenance of the paper's SAE
features. See [current validation and limitations](../../../../docs/REPRODUCTION.md).
Set `HF_DATA_FASTA` explicitly to the intended training split; automatic largest-file
selection and fallback checkpoint substitution have been removed.

All scripts are under `examples/notebooks/interpretability_analysis/sae_repro_release/`.

## 1) Environment

### Option A: Local Python

```bash
cd /path/to/EVA1
pip install -r examples/notebooks/interpretability_analysis/sae_repro_release/requirements-sae-training.txt
export EVA_SAE_LOCAL=1
```

### Option B: Docker (recommended)

The `.sh` scripts default to a running `eva-repro` container with source mounted
at `/eva`. Start it with the main reproduction guide, then run from the host:

```bash
bash examples/notebooks/interpretability_analysis/sae_repro_release/scripts/run_all_smoke.sh
```

Override the container name only if needed:

```bash
CONTAINER_NAME=eva-repro bash examples/notebooks/interpretability_analysis/sae_repro_release/scripts/run_all_smoke.sh
```

If you need a dedicated image instead, build from repo root:

```bash
cd /path/to/EVA1
docker build \
  -f examples/notebooks/interpretability_analysis/sae_repro_release/Dockerfile \
  -t eva-sae:latest \
  .
```

Run container:

```bash
docker run --gpus device=0 --name eva-repro --rm -it \
  -v "$PWD":/eva \
  -w /eva \
  eva-sae:latest
```

## 2) Download model + dataset with pinned revisions

Use fixed commit hashes from Hugging Face to avoid drift.

```bash
export HF_MODEL_REVISION=514db6705637c1ec963b728768fc9b34728699ee
# Set HF_DATA_REVISION from the manifest for the exact intended split.
: "${HF_DATA_REVISION:?Set the verified dataset revision before downloading}"

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
- Required explicit input: `export HF_DATA_FASTA=/path/to/train.fa`
- Record its SHA256 and training/validation split provenance before running.
- Optional checkpoint override: `export SAE_CKPT_DIR=/path/to/checkpoint_dir`

## 4) Validate environment and paths

```bash
python examples/notebooks/interpretability_analysis/sae_repro_release/scripts/validate_env.py

bash examples/notebooks/interpretability_analysis/sae_repro_release/scripts/check_hf_paths.sh batch_topk
bash examples/notebooks/interpretability_analysis/sae_repro_release/scripts/check_hf_paths.sh sae_l1_penalty
```

## 5) Smoke test (fast sanity check)

```bash
bash examples/notebooks/interpretability_analysis/sae_repro_release/scripts/run_all_smoke.sh
```

## 6) Full training

```bash
bash examples/notebooks/interpretability_analysis/sae_repro_release/scripts/run_batch_topk_full.sh
bash examples/notebooks/interpretability_analysis/sae_repro_release/scripts/run_sae_l1_penalty_full.sh

# or run both sequentially
bash examples/notebooks/interpretability_analysis/sae_repro_release/scripts/run_all_full.sh
```

## Output locations

- Checkpoints: `examples/notebooks/interpretability_analysis/sae_repro_release/outputs/*/checkpoints`
- Logs: `examples/notebooks/interpretability_analysis/sae_repro_release/logs`

## Reproducibility controls

- Global random seed is set for `random` / `numpy` / `torch`.
- Deterministic flags are enabled by default in configs (`deterministic: true`).
- Full configs and smoke configs are both versioned under `configs/`.

## Historical validation claim (not independently verified here)

- Smoke validation was executed on April 9, 2026 in GPU Docker runtime (`eva:latest`).
- Both modes (`Batch-TopK SAE`, `sae_L1_penalty`) completed smoke training and produced checkpoints.

## Common errors

- `HF_DATA_FASTA not found`: wrong `HF_DATA_ROOT` or missing FASTA files.
- `Invalid SAE_CKPT_DIR`: checkpoint directory missing `config.json` or `model_weights.pt`.
- `Import failed: megablocks...`: install dependencies again or use Docker image.
