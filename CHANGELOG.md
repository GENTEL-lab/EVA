# Changelog

All notable changes to EVA are documented in this file.

The project uses Git tags for code releases. A reproducible release should also record the corresponding Hugging Face model revision, dataset revision, and Docker image or build command.

## v1.1.0

### Fixed

- `docker build -f docker/Dockerfile` failing on a fresh machine. Three
  independent blockers, all now resolved:
  - The base image tag `pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime-ubuntu22.04`
    does not exist on Docker Hub; the image is now built from
    `pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel`, which additionally
    provides the nvcc toolchain required to build `flash-attn`,
    `grouped-gemm`, `stanford-stk` and `megablocks` from source (none of
    them publish prebuilt wheels on PyPI, so the previous runtime-only
    base could never compile them).
  - `docker/requirements.txt` pinned packages that cannot be installed
    from PyPI: `python-apt==2.4.0+ubuntu4` (Ubuntu archive only) and the
    `+cu124` local versions of `torch`/`torchvision` (provided by the base
    image instead). Unused system-package noise from the legacy
    `pip freeze` (`ovs`, `pandoc`, `netifaces`, `python-snappy`,
    `sentry-sdk`-unrelated entries, `accelerate`, `psutil`, ...) was
    removed and the file normalized to LF line endings. Every remaining
    pin was verified resolvable on PyPI.
  - `.dockerignore` excluded `docker/` from the build context, breaking
    `COPY docker/requirements.txt`. `fig/` (34 MB, not part of the image)
    is now excluded to keep the build context small.
- CUDA source builds are now deterministic: `TORCH_CUDA_ARCH_LIST` is fixed
  to `7.5;8.0;9.0` (Tesla T4, Ampere/Ada via binary compatibility, Hopper)
  and `triton`/`numpy`/`packaging` are pre-pinned before installing
  `megablocks` (whose loose `triton>=2.1.0` constraint would otherwise pull
  a newer Triton than the released environment).
- Triton failing with `ImportError: cuda_utils.so: undefined symbol:
  cuModuleGetFunction` on Singularity/Apptainer clusters. Released images
  contained 0-byte `libcuda.so.*` placeholders inherited from
  `docker commit`-based builds; Docker `--gpus` masked them while
  Singularity `--nv` left them visible to Triton's link step. Fixes:
  - New runtime self-healing module `eva/_runtime_env.py`, invoked on
    `import eva` and by the container entrypoint: it stays silent when
    Triton's default ldconfig-based discovery works (regular Docker) and
    otherwise locates the real driver library, exports
    `TRITON_LIBCUDA_PATH` (honoured with highest priority by Triton 3.1.0),
    falls back to a writable `TRITON_CACHE_DIR` when `$HOME` is read-only,
    and purges mis-linked cached Triton helper libraries.
  - `docker/Dockerfile` removes driver-library placeholders at build time
    and clears the stub-only `LIBRARY_PATH` inherited from the base image.

### Added

- `docker/entrypoint.sh` as the container entrypoint: exports the Triton
  driver environment (Docker `--gpus` and Singularity `--nv`) before
  handing off to the NVIDIA entrypoint / user command.
- `docker/EVA.def`: Singularity/Apptainer definition file to build a `.sif`
  directly from the local Docker image, with placeholder cleanup in `%post`.
- `docker/smoke_test.sh`: pre-release smoke test that probes the Triton
  driver under both Docker `--gpus` and Singularity `--nv`.
- README sections for HPC/Singularity usage, Triton troubleshooting, and
  the pre-release smoke test.

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
