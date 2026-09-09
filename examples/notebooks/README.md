# Analysis notebooks

Use this directory to explore results interactively, inspect intermediate
outputs and generate plots. Notebook folders also contain supporting scripts
and data; the directory is organized by research task, not by file extension.

| Task | Start here |
|---|---|
| Benchmark summaries and metrics from your own predictions | [Benchmark notebook](prediction/benchmark_reproduction.ipynb) |
| RNA generation tutorial | [Generation notebook](generation/generation_tutorial.ipynb) |
| Interpretability analysis | [Analysis guide](interpretability_analysis/README.md) |
| Design case studies | [Design notebooks](design/) |

## How this relates to reproduction

Notebooks and command-line tools are two ways to work through a computation.
They share scripts and resources rather than defining separate benchmarks.
For example, `prediction/benchmark_reproduction.ipynb` calls
[`scripts/benchmark_release.py`](../../scripts/benchmark_release.py), which reads
the manifests and reference tables in
[`examples/reproduction/benchmark_release/`](../reproduction/benchmark_release/).
That notebook performs CPU analysis; fresh model inference has its own command.

[`examples/reproduction/`](../reproduction/README.md) holds fixed inputs, protocols,
reference outputs and dated records, together with preserved historical code.
Data and helper scripts live beside their notebooks. Both directories
participate in the workflows listed in the [examples index](../README.md).

Start with the [reproduction guide](../../docs/REPRODUCTION.md) for a complete
workflow, or [benchmark protocols](../../docs/BENCHMARK_PROTOCOLS.md) to evaluate
predictions from a model you run in your own environment.
