# Validation and release status

The integrated **1.2.0rc1** source is public on `main`. The reproducibility
repairs were merged at `9b5d25ac7cb26de88420f56021f4c0eeb2052e9e`.
A formal release for this revision and an archival code DOI are pending;
the older `v1.1.1` release does not contain these repairs.

## Recorded validation: September 9, 2026

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

`reproduction/release/clean_validation.json` records the commit and hashes.
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
`--strict-reference` diagnostic. [Expected outputs](../reproduction/milena_14b/expected/README.md)
record the measurements. Historical batch-size observations are context, not
a demonstrated cause of the difference.

## Remaining coverage and publication work

- Recovered comparison-model hashes are in the [resource index](OFFICIAL_MODEL_RESOURCES.md).
  Public downloads, original runtimes or producers remain missing for some methods;
  cached artifact identities alone do not close those gaps.
- Training machinery is tested at small scale. Original dense-model code and
  some paper-specific training/run bindings are not available.
- Some figure-specific input/command mappings and complete SAE generation
  outputs remain outside the bundled reproducible workflows.
- Formal release, code DOI and the corresponding manuscript citation remain
  separate steps. No placeholder DOI is provided.

The [six reviewer requirements](REVIEWER_REQUIREMENTS.md) and
[paper workflow table](PAPER_WORKFLOWS.md) identify the evidence and remaining
scope. Historical notes are dated background records; use current guides
for present software behavior.
