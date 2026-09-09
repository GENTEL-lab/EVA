# Reproducibility release candidate

This is an integrated local candidate based on public commit
`07f4cb523ed5053f1ac11d79e4c524167b00bbdd`. It has not been published or assigned
a code DOI. It combines the existing engineering fixes with the benchmark
supplement and a pinned EVA-1.4B Milena entry point.

The current submission data are preserved as confirmed by the author. Older
provenance reports are retained as dated evidence and do not automatically
establish an error in replacement data.

The integrated test suite passed 128 tests. Synthetic pretraining, mid-training
and fine-tuning each completed two optimizer steps with exact saved-weight
reloads. The three notebook code cells executed in an independent kernel;
its optional custom prediction section remains unconfigured. `pip check` passed.
These checks ran in a new container using an existing CUDA base image with
explicitly installed dependencies. A separate build from the official PyTorch
base is being validated and is not implied by these execution checks.

The complete 135-row public-checkpoint Milena inference ran successfully:
Spearman 0.8394237924835843 versus 0.8360456283218484 in the current table.
All 135 new scores are identical to a separate earlier fresh run under different
PyTorch/MegaBlocks versions. The historical score difference remains unexplained.
See `reproduction/milena_14b/expected/README.md` for the observed result and
cached-checkpoint versus online-download distinction.

Release still requires a complete public-checkpoint example with interpretable
numerical agreement and accurate documentation of all required workflow resources.
A failed or unexplained benchmark comparison prevents a claim that the
reviewer's reproducibility concerns have all been resolved.

Exact model and input identities are in `reproduction/milena_14b/manifest.json`.
See [PAPER_WORKFLOWS.md](PAPER_WORKFLOWS.md) for missing competitor-resource
bindings and workflow limitations. No placeholder DOI is provided.
