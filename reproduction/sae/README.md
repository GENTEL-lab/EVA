# Archived SAE reproduction inputs

This directory preserves historical EVA steering inputs and source files. Run
`python scripts/reproduce_sae.py --output /path/to/new/output --plot` from the
repository root. See `docs/SAE_REPRODUCTION.md` for the exact analysis contracts.

The archived scripts are evidence, not portable entry points: they retain their
original absolute paths. The portable entry point does not import them as modules.
No model weights, training FASTA, or author credentials belong in this bundle.

The original likelihood result is a selected best-condition summary. The eight
structure examples are author-selected illustrations, not an unbiased success
rate. Table S30 can be rebuilt from its archived aggregate table; that operation
is not a complete raw-record recomputation of 213/225.
