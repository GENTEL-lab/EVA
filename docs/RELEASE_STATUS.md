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
The complete source archive was then built from the official PyTorch
2.5.1/CUDA 12.4 base (resolved image digest
`sha256:14611869895df612b7b07227d5925f30ec3cd6673bad58ce3d84ed107950e014`).
In this clean image, all 128 tests, the three installed CLI help commands,
training/save/reload, the full 135-row inference, archived SAE recalculation,
summary audit and independent-kernel notebook execution completed. `pip check`
reported no broken requirements. The build used the Aliyun PyPI mirror with
the pinned dependencies; EVA attention uses torch SDPA, so no external
flash-attn package is installed. GPU work used one idle A100 and batch size 1.

The complete 135-row public-checkpoint Milena inference ran successfully:
Spearman 0.8394237924835843 versus 0.8360456283218484 in the current table.
All 135 new scores are identical to a separate earlier fresh run under different
PyTorch/MegaBlocks versions. The recovered original log records batch size 45; its causal role in the
score difference has not been established.
See `reproduction/milena_14b/expected/README.md` for the observed result and
successful public download and byte-identical GPU cache verification.

Release still requires a complete public-checkpoint example with interpretable
numerical agreement and accurate documentation of all required workflow resources.
A failed or unexplained benchmark comparison prevents a claim that the
reviewer's reproducibility concerns have all been resolved.

Exact model and input identities are in `reproduction/milena_14b/manifest.json`.
See [PAPER_WORKFLOWS.md](PAPER_WORKFLOWS.md) for missing competitor-resource
bindings and workflow limitations. No placeholder DOI is provided.
