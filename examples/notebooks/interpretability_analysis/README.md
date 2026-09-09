# Interpretability Analysis

This directory contains:

- `interpretability.ipynb`: main notebook for interpretability figures/analyses.
- `figures/`: generated paper figures.
- `intermediate_data/`: cached arrays/json used by the notebook.
- `sae_repro_release/`: standalone reproducibility bundle for SAE training
  (`Batch-TopK SAE` and `sae_L1_penalty`).

Path behavior:

- Default paths are resolved relative to the EVA1 repository root.
- The notebook no longer depends on launching Jupyter from
  `examples/notebooks/interpretability_analysis`.
- `reproduce_paths.py` resolves bundled data/figure/font paths from the repo
  layout under `EVA1/`.

For SAE training reproduction, start from:

`examples/notebooks/interpretability_analysis/sae_repro_release/README.md`
