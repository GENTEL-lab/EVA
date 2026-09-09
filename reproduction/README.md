# Reproduction resources

This directory keeps the resources that identify a computation: input and model
manifests, scoring protocols, reference results, file hashes and dated validation
records. It also contains the portable comparison-model runner and preserved
historical implementations. It is not a second set of analysis notebooks.

| Folder | Contents |
|---|---|
| [milena_14b/](milena_14b/) | Fixed EVA checkpoint/input identities and expected Milena outputs |
| [benchmark/](benchmark/) | Comparison-model CLI, fixtures, protocols and historical scoring sources |
| [benchmark_release/](benchmark_release/) | Benchmark reference tables and evaluation manifests used by scripts and notebooks |
| [essentiality/](essentiality/) | Inputs, archived results and provenance for essentiality calculations |
| [sae/](sae/) | SAE protocols, manifests and archived analysis resources |
| [release/](release/) | Dated software validation and model-resource identity records |

## How the folders work together

- [`notebooks/`](../notebooks/README.md) provides interactive analysis and plots,
  with supporting data and helpers beside the notebooks.
- [`scripts/`](../scripts/) provides command-line entry points for repeatable runs.
- This directory supplies fixed resources and reference evidence for those runs.

For example, `scripts/reproduce_milena.py` uses the manifest here, reads the assay
inputs under `notebooks/prediction/data/`, and calls the preserved scoring
implementation under `benchmark/`. Its output is written to the directory you
select. The [task guide](../docs/REPRODUCTION.md) documents the complete command.

Files under `upstream/` and `archive/` preserve historical versions, while
current commands are listed in the task guides. These folders retain their
existing paths because manifests and scripts refer to them.
