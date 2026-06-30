# Changelog

All notable changes to EVA are documented in this file.

The project uses Git tags for code releases. A reproducible release should also record the corresponding Hugging Face model revision, dataset revision, and Docker image or build command.

## v1.0.0

### Added

- Standard Python package metadata through `pyproject.toml`.
- Editable source installation with `python3 -m pip install -e .`.
- Command-line entry points:
  - `eva-generate`
  - `eva-predict`
  - `eva-evolve`
- Package data inclusion for the bundled tokenizer and lineage table.
- Release documentation in `README.md`.

### Changed

- Docker build instructions now use the repository root as the build context.
- Docker shell wrappers infer the repository root instead of requiring a hard-coded host path.
- Model loading no longer requires installed package directories to be writable.

### Notes

- Large checkpoints and datasets are not part of the Python package.
- Full MoE inference requires the CUDA/GPU runtime dependencies described in `docker/requirements.txt`.
- Release artifacts such as `dist/`, wheels, and source tarballs should be generated during release and not committed.
