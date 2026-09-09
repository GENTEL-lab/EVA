# Validation and release status

**v1.2.0** packages the reproducibility repairs and repository layout updates.
The source archive and its checksums are publicly available on
[Figshare](https://doi.org/10.6084/m9.figshare.33490096.v3) and in the GitHub release
assets. The archive contains all 753 tracked files at tag v1.2.0, commit
`9e8d63bbe4ab4a0c5cf2646164751b52e8a71462`. The public source ZIP matches the release
asset (SHA256 `e1441561c552b308e82f274a7fddf510484beac109e6fd1dbe305ad76ae0bc7b`).
Figshare version 3 corrects archive metadata and the licence to Apache-2.0;
the source ZIP is unchanged. Use the DOI above when citing the fixed source,
whose embedded citation metadata predates the final deposit.
The [GitHub release](https://github.com/GENTEL-lab/EVA/releases/tag/v1.2.0)
and `CITATION.cff` identify this software version. Models and datasets retain
their separately documented version identifiers.

## Repository layout validation: September 9, 2026

Commit `780f226` passed 146 regression tests and Python 3.10/3.11 CI. The
relocated benchmark notebook, CPU example, wheel installation outside the
checkout and container source installation passed. Both Milena modes completed
all 135 inputs, and pretraining, mid-training and fine-tuning checks verified
parameter updates and exact checkpoint reloads. The v1.2.0 release changes only
version and citation metadata and release documentation relative to that commit.

## Usability update validation: September 9, 2026

The updated Milena entry point was run twice on an idle A100-SXM4-80GB,
with the existing pinned runtime, one visible GPU and batch size 1. Each run
scored all 135 inputs, produced metrics and plots, and exactly matched the
saved fresh-prediction vector. Default mode returned **0**; the optional
strict-reference mode returned **2**. Scoring took about 12 seconds per run;
peak PyTorch-allocated GPU memory was 2.94 GiB.
The [dated reports](../examples/reproduction/release/usability_gpu_20260909/README.md)
record the source commit, file hashes, environment and both execution modes.

The expanded suite passed **139 tests** without skips in the existing Docker
runtime. Current Python 3.10/3.11 CPU jobs, source-archive execution, wheel
installation outside the checkout and CLI checks are available in
[GitHub Actions](https://github.com/GENTEL-lab/EVA/actions/workflows/cpu.yml).
The clean CPU environment also exposed an eager GPU-backend import in the
fine-tuning entry point; that import now occurs only during model construction.

## Earlier clean source-build validation: September 9, 2026

The source archive at `70c6c4f7aff2cf12210d8b280c1d4569ca22c0a0` was built in a
clean Docker environment. The subsequent `9b5d25a` commit added documentation
and validation records without changing runtime code or inputs.

| Check | Recorded result |
|---|---|
| Regression tests | 128 passed; none skipped or failed |
| Installation | `pip check` and all three installed CLI help commands passed |
| Synthetic training | Two steps per stage; pretrain/midtrain/fine-tune weights saved and reloaded exactly |
| Notebook | Three code cells executed in an independent kernel; optional custom predictions unconfigured |
| Milena inference | All 135 samples scored from the pinned public checkpoint |
| Archived calculations | Milena metric, SAE calculations and benchmark summary audit executed |

`examples/reproduction/release/clean_validation.json` records the commit and hashes.
The image used PyTorch 2.5.1/CUDA 12.4, Transformers 4.55.0 and MegaBlocks 0.7.0.
The base-image digest was
`sha256:14611869895df612b7b07227d5925f30ec3cd6673bad58ce3d84ed107950e014`.
Installation used the Aliyun PyPI mirror with pinned versions. GPU validation
used one idle A100 and batch size 1. This historical record does not automatically
validate later changes; current CPU checks run separately in GitHub Actions.

## Representative benchmark result

New Milena inference gives **0.8394237924835843**, compared with the stored
**0.8360456283218484**. The **0.0033781641617359748** difference is accepted for
this representative example and is not a merge or release blocker. Both
numbers and all predictions remain available. No input labels, scoring
objective or paper metric have been changed to reduce the difference.

[Reproduction](REPRODUCTION.md) documents normal completion and the optional
`--strict-reference` diagnostic. [Expected outputs](../examples/reproduction/milena_14b/expected/README.md)
record the measurements. Historical batch-size observations are context, not
a demonstrated cause of the difference.

## Validation scope

- Recovered comparison-model hashes are in the [resource index](OFFICIAL_MODEL_RESOURCES.md).
  Local checkpoints and user-generated predictions are supported. Exact paper
  runtimes, scoring implementations or checkpoint bindings remain unverified for
  some methods; file identities alone do not establish those correspondences.
- Training machinery is tested at small scale. Original dense-model code and
  some paper-specific training/run bindings are not available.
- Some figure-specific input/command mappings and complete SAE generation
  outputs remain outside the bundled reproducible workflows.

The [six reviewer requirements](REVIEWER_REQUIREMENTS.md) and
[paper workflow table](PAPER_WORKFLOWS.md) identify the evidence and remaining
scope. Historical notes are dated background records; use current guides
for present software behavior.
