# Examples

Tutorials and paper benchmarks share one home. Choose a task below; its guide
connects the notebook or command to the required data, configuration and outputs.

| Task | Start here |
|---|---|
| Recalculate a benchmark or run EVA on the complete Milena assay | [CPU and GPU walkthroughs](../docs/REPRODUCTION.md) |
| Analyze benchmark tables or evaluate your own predictions | [Benchmark notebook](notebooks/prediction/benchmark_reproduction.ipynb) · [Local-model evaluation](../docs/BENCHMARK_PROTOCOLS.md#using-your-own-model-outputs) |
| Explore RNA generation and design | [Generation tutorial](notebooks/generation/generation_tutorial.ipynb) · [Design notebooks](notebooks/design/) |
| Explore model interpretation | [Interpretability guide](notebooks/interpretability_analysis/README.md) |
| Configure a CLI task | [Example YAML files](config/) · [Sample inputs](data/test_data/) · [CLI reference](../docs/USAGE.md) |

`notebooks/` contains interactive analyses and their supporting files.
`reproduction/` contains the fixed protocols, reference data and historical
sources those analyses use. They support the same reproduction workflows:
for example, the benchmark notebook calls `scripts/benchmark_release.py` from
the repository root and reads `reproduction/benchmark_release/` here.

Run documented shell commands from the **repository root**. Notebook paths are
resolved by the notebook or relative to its own folder, as shown in its setup
cell. Save new results to the output directory you choose.

For pretraining and fine-tuning, start with [training](../training/README.md).
