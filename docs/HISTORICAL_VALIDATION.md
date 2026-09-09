# Historical validation record

The following records describe earlier snapshots. They are retained as evidence,
not as findings about replacement data. For the current candidate see [REPRODUCTION.md](REPRODUCTION.md).

# Reproduction and validation boundaries

This source repair distinguishes **software execution**, **numerical agreement**
and **paper-artifact provenance**. A successful command does not establish all three.
The baseline public code is commit `07f4cb523ed5053f1ac11d79e4c524167b00bbdd`.
Repairs in this workspace are uncommitted and have not been pushed or released.

## Recovered original analysis workflows

- [Historical benchmark and competitor protocols](BENCHMARK_PROTOCOLS.md):
  model/config/tokenizer identity, archived prediction arithmetic, label
  conflicts, parameterized RNA-MLM/protein adapters, and the supplemental
  135-sequence historical 1.4B inference with recorded numerical differences.
- [Essentiality and position ablation](ESSENTIALITY_REPRODUCTION.md):
  full archived prediction recomputation, stable identities, separate 5%/50%
  fractions and original/corrected GC baselines.
- [Historical SAE results](SAE_REPRODUCTION.md):
  original selected 401/401 likelihood analysis, eight fixed author-selected
  examples, archived Table S30 reconstruction, and a separate historical
  1.4B/layer-13 single-case inference matching the archived sample.

The full-source bundle carries original files and SHA-256 manifests under
reproduction/. These are not included in the wheel. A successful archived
summary calculation is not a new model-inference run.

## Environment and installation

The tested runtime is Python 3.11, PyTorch 2.5.1+cu124, Transformers 4.55.0,
NumPy 2.0.2, FlashAttention 2.6.3, MegaBlocks 0.7.0, SciPy 1.14.1 and
ViennaRNA 2.7.0. Use a CUDA GPU with sufficient available memory; expose only
one GPU on a shared server. CUDA_VISIBLE_DEVICES inside Docker uses container indices.

```bash
# From a complete checkout, build the published recipe when no runtime exists.
docker build -f docker/Dockerfile -t eva-repaired:local .
mkdir -p checkpoint results
docker run --rm -it --gpus device=0 --name eva-repro \
  -v "$PWD":/eva -w /eva eva-repaired:local bash
```

If reusing an existing compatible runtime, install only missing Python extras
in an isolated environment (do not update another job's environment):

```bash
python -m pip install -e '.[benchmark,design,dev,notebook]'
python -m pytest -q tests
```

Editable/wheel installs include `eva` and `tools` and their CLI commands. Training,
notebooks, configs and benchmark datasets require a **full source checkout**.
`pip install -e .` does not install the compiled FlashAttention/MegaBlocks stack.
Its matched CUDA build commands are in `docker/Dockerfile`; `--help` is not a
GPU inference test. A working pre-existing image is not proof that a clean image
build has completed. Record those checks separately.

Training evaluation analysis (`training/eval/scripts/analyze_results.py`) also
requires pandas, seaborn and scikit-learn; the PyTorch base image does not supply
them. The Docker recipe pins pandas 2.2.3, seaborn 0.13.2 and scikit-learn 1.5.2,
with their small additional runtime dependencies. For an existing compatible
isolated environment use `pip install -e '.[notebook]'` before running this
analysis entry point. Its `--help`, checkpoint-metrics JSON/CSV loading and basic
statistics are regression-tested; this is analysis of saved metrics, not a new
model evaluation. Compiled FlashAttention/MegaBlocks/CUDA components still follow
the separate Docker build instructions above. Third-party competitor runtimes
and checkpoints (for example ERNIE/fairseq, Evo2, CodonFM and ESM) are not supplied
by the notebook extra; use the model-specific requirements and limitations in
[BENCHMARK_PROTOCOLS.md](BENCHMARK_PROTOCOLS.md).

## Pinned checkpoint and representative full assay

```bash
python scripts/download_reproduction_checkpoint.py --model EVA_21M --destination checkpoint
python scripts/reproduce_dms.py \
  --checkpoint checkpoint/EVA_21M \
  --output results/milena_21m --batch-size 1
```

The downloader fixes HF model revision
`514db6705637c1ec963b728768fc9b34728699ee` and verifies the weight SHA256:
`45d56a7399c4936429da149edf5003bbb5999490a64ec7d793bf5ac98116eb01`.
It writes a manifest including config/tokenizer hashes. `checkpoint/` itself is
not a valid model folder: pass the `EVA_21M` subdirectory. Loading never modifies
the installed tokenizer; ambiguous multiple-weight directories fail explicitly.

The released Milena_2021_cata bundle contains 135 sequences and 135 positional
labels. Inputs are its FASTA and `intensities` JSON in
`notebooks/prediction/data/ncRNA`.
The script checks counts, unique identifiers, finite values and canonical RNA;
it logs hashes and original row order. Hashing identifies the supplied files,
but cannot independently prove that positional labels were originally paired correctly.

Scoring normalizes T to U, forms `<bos>5SEQUENCE3<eos>`, and averages causal
log-likelihood over nucleotide targets, excluding `5`, `3`, `<eos>`. The optional
`--rna-type` adds `|<rna_TYPE>|` after BOS; those prefix targets are **not** excluded
by the existing worker. Do not switch conditioning or token masks to fit a metric.
This protocol is reconstructed from a located legacy adapter. The public EVA-21M
weight, config and tokenizer are byte-identical to the recovered historical
checkpoint, but the exact invocation and sequence-to-label inputs for the paper
table row remain unverified. `--reference-model` selects only the comparison CSV
row: it does not change the checkpoint, conditioning or mean reduction. For the
recorded historical 1.4B sum protocol use the dedicated runner documented in
[BENCHMARK_PROTOCOLS.md](BENCHMARK_PROTOCOLS.md), not this flag alone.

Outputs: `predictions.csv` (IDs, sequence, labels, scores) and `report.json`
(Spearman, comparison, versions, source/input/weight hashes, elapsed time and
peak allocated GPU memory). Existing output directories are refused.
Exit 0 means agreement within the stated threshold; exit 2 means complete scoring
but disagreement. Default comparison uses half the last published decimal place,
not an arbitrary 0.001 tolerance. An explicit `--tolerance` is diagnostic and
must not be interpreted as proof of historical reproduction.

To draw a diagnostic scatter plot with a Spearman summary from saved predictions:

```bash
python scripts/plot_dms_predictions.py \
  --input results/milena_21m/predictions.csv \
  --output-directory results/milena_21m_plot
```

This does not run the model or change predictions/labels. Its PNG/SVG and report
are diagnostics of the supplied records, not a reconstruction of Figure 2 or a
resolution of the original sequence-to-readout provenance.

**Observed result:** released EVA-21M gives Spearman **0.9037308669414685**;
the archived EVA-21M table gives **0.5074679659247847**. This is a completed
end-to-end run with a **numerical mismatch**, not a reproduced paper result.
Resolve the original scoring invocation, FASTA/label pairing and table-generation
provenance before making a rebuttal claim of numerical reproduction.

The separate historical 1.4B audit recovered the stored-score arithmetic
(rho=0.8360456283218484) and completed a fresh 135-sequence inference with the
recovered native model/worker. The new vector was not bit-exact at any row;
its rho against the same unresolved positional labels is 0.8394237924835843.
The original sequence/readout linkage, inference invocation and immutable
score-to-weight binding remain unresolved. See the benchmark guide for distinct
archive-only and fresh-inference commands; these results do not replace the
EVA-21M representative-assay discrepancy.

## Training, generation and SAE smoke tests

```bash
python scripts/smoke_workflows.py --checkpoint checkpoint/EVA_21M --output results/workflow_smoke
python tools/generate.py --checkpoint checkpoint/EVA_21M --format clm \
  --num_seqs 1 --max_length 32 --min_length 1 --output results/generated.fa
python tools/directed_evolution.py --checkpoint checkpoint/EVA_21M \
  --input tests/data/train_smoke.fasta --iterations 1 --mutations 1 \
  --beam_width 2 --output_count 1 --output results/evolved.fa
```

The workflow test exercises real trainer setup, forward/backward, two optimizer
steps and exact DCP/PT weight round trips. Tiny synthetic training configs are
not paper recipes. It also exercises both SAE trainers on released EVA-21M
activations (layer 1, not the manuscript layer 13), saves and reloads SAE weights.
See the [training guide](../training/pretrain/README.md) and
[aptamer guide](../finetune/aptamer/script/README.md).

Pretraining `output_token_mask: legacy_generation` retains the historical CLM
mask; mixed/completion training must explicitly choose `output_token_mask: none`
because the CLM mask excludes GLM end tokens. This choice is recorded in smoke
configs and must be verified against original paper configurations separately.
Sample processing errors no longer silently change GLM to CLM or insert dummy rows.

SAE shell wrappers use a running `eva-repro` container, or `EVA_SAE_LOCAL=1` for
an already-configured local environment. Always specify the exact FASTA split
and checkpoint; no largest-file choice or model substitution is performed.
Missing dependencies, non-finite predictions/losses, failed hidden extraction,
and zero SAE optimizer steps are errors, not successful reduced workflows.

Separate from these tiny training checks, the original 1.4B/layer-13 SAE and
base checkpoint were located and hash-recorded. The historical single-case
helper in `reproduction/sae/run_historical_smoke.py` reproduced the fixed Pepper
case's unsteered/0x generated sequences, RNAfold structures and scores exactly,
with the archived seed/context and 238 final state tensors checked. This does
not establish full-cohort reproduction or correspondence to every public weight
release. Historical resources and the full argument/checkpoint bindings are
described in [SAE_REPRODUCTION.md](SAE_REPRODUCTION.md).

## Coverage and remaining provenance gaps

| Workflow | Source entry / verified evidence | Boundary |
|---|---|---|
| RNA DMS | `tools/predict.py`, `scripts/reproduce_dms.py`, `scripts/reproduce_historical_benchmark.py`, `reproduction/benchmark/run_historical_14b.py` | 21M full assay executed with table mismatch; 1.4B archived arithmetic recovered and full fresh inference completed with vector differences. Original labels and exact historical invocation remain unresolved. |
| Protein DMS | `tools/predict.py --mode protein`; recovered RNA reverse-translation/ESM workers under `reproduction/benchmark/upstream/protein/`; strict `reproduction/benchmark/run_competitor.py protein-esm` launcher | EVA reverse translation is RNA surrogate scoring, not direct amino-acid likelihood. Native adapter source and strict input checks are available; exact per-assay codon/conditioning/reference and checkpoint linkage, ESM ensembles, and a full protein forward benchmark remain incomplete. |
| Pre/mid-training and finetuning | `training/`, `finetune/`, `scripts/smoke_workflows.py`; five recovered run configurations in the benchmark bundle | Real synthetic forward/backward/save/reload checks passed. The observed mixed CLM/GLM and EP/DP configurations apply to identified runs; full training data/release lineage and the manuscript's stage-conditioning account still require reconciliation. |
| Dense-model training | historical `model_dense` imports | Architecture source missing; explicitly unsupported |
| Essentiality and position ablation | `scripts/reproduce_essentiality.py`, `reproduction/essentiality/source_manifest.json`; [ESSENTIALITY_REPRODUCTION.md](ESSENTIALITY_REPRODUCTION.md) | All ten stored position-ablation groups (95,538 CDS records each) recomputed on CPU; 5%/50% summary-key defect and record/sequence GC join corrected. This is not fresh model inference. Main dataset-building/inference producer, source labels and full score-to-weight bindings remain unresolved. |
| Generation / optimization | `tools/generate.py`, `tools/directed_evolution.py` | Engineering checks do not reproduce wet-lab or structural validation |
| SAE archived analyses | `scripts/reproduce_sae.py`, `reproduction/sae/`; [SAE_REPRODUCTION.md](SAE_REPRODUCTION.md) | Raw stored likelihood rows reproduce the original selected 401/401 summaries/statistics; eight author-selected examples verified; Table S30 CSV-to-LaTeX reconstruction exact. Selection is not an unbiased intervention success rate, and 213/225 has not been rebuilt from a complete raw-generation cohort. |
| Steering | `tools/sae_steering.py` for engineering use; `reproduction/sae/run_historical_smoke.py` for the recovered historical case | Tiny 21M/layer-1 smoke and one real 1.4B/layer-13 Pepper case are separately validated. Original feature/seed/context/checkpoint evidence exists for that case. Cohort-level inference, some original generation arguments, and historical-to-public tensor correspondence remain incomplete; 0x is ablation, not baseline. |
| Competitor models | Recovered RNA/protein workers under `reproduction/benchmark/upstream/`; `reproduction/benchmark/run_competitor.py` for bounded RNA-MLM and strict native ESM execution/metrics | Real tiny random RNA-MLM machinery test and protocol/input checks passed, not released competitor-model benchmarks. Model-specific runtimes/weights and paper linkage remain required; some external producers, including ProGen3 scoring, are still missing. Plotting or archived metric recalculation is not evaluation. |

The updated 49-label coverage inventory is maintained in the delivery evidence
alongside this local repair; the earlier nine-candidate inventory is historical,
not the complete current source recovery. Current manifests cover 95 benchmark
source/data files plus 42 historical 1.4B files, 110 SAE source/data files and the
essentiality inputs/original scripts. Original archived scripts retain historical
paths and defects for provenance; use the documented portable entry points, not
an arbitrary archived script. The portable steering loader reads saved `cfg.k`
and removes hooks on failure. Manuscript code DOI and the final paper-specific
public release are still pending; no DOI or complete-release claim is fabricated.

## Additional steering smoke command

Install `.[steering]` if Matplotlib is absent. The example below uses the newly
trained tiny SAE, arbitrary diagnostic features 0/1 and constant clamp values.
It validates the intervention pipeline, not the manuscript's biological features.

```bash
python tools/sae_steering.py --checkpoint checkpoint/EVA_21M \
  --sae results/workflow_smoke/batch_topk/checkpoint_step2.pt --layer 1 \
  --features-a 0 --features-b 1 --scales 0,1 --scale-source constant \
  --clamp-value 1 --prefix ACGUACGU --target ACGU --suffix ACGUACGU \
  --out-prefix results/steering_smoke
```

The observed OpenRNA repository revision is
`103c79aab4d625828721a25ab98be51763c326bb`. It identifies the repository snapshot,
not the original manuscript training split. The representative DMS run uses
the hashed assay files in the code repository, not OpenRNA training data.

## Delivery and clean-image acceptance

The final source ZIP is independently reconstructed from the recorded Git
baseline plus `eva_repair.patch`; `repair_manifest.json` and
`clean_delivery_verification.json` record its source hashes and ZIP checks.
These artifacts and workflow logs are supplied alongside the source ZIP under
`code_revision/validation_20260905_round2/`. The final existing-image CPU suite is
recorded separately in `full_regression_final.log`; consult its exact test count
and outcomes, including the real analysis dependency/JSON/CSV checks.

A new Docker image has a separate acceptance gate: confirm the completed build
log, exact `docker image inspect` identity, installed dependency versions,
`pip check`, required entry points and pytest results **for that final image and
source snapshot** in the delivery's `docker/` evidence. Check the actual recorded
exit codes and image identity; this guide does not assert that those checks have
passed. Missing or incomplete evidence means that gate is not established.
Existing-image inference, wheel installation, source/ZIP integrity and paper
numerical reproduction are different validations and cannot substitute for one
another. Build progress or a previous download-stage failure is not a permanent
description of the delivered environment.
