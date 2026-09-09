# Installation

Use the complete source checkout for paper workflows. The Python wheel includes
`eva` and the inference CLIs; training scripts, notebooks and benchmark inputs
are source-checkout resources.

## Choose an environment

| Task | Environment | What is verified |
|---|---|---|
| Recalculate the bundled Milena metric | Python 3.10 or 3.11; standard library only | No model or GPU needed; [CPU example](REPRODUCTION.md#cpu-recalculate-stored-predictions) |
| Package/CLI use and CPU development tests | Python 3.10 or 3.11; editable install with the required extras | Package installation and CPU tests; not full MoE inference |
| Model inference and training | Linux, NVIDIA GPU and the Docker runtime below | PyTorch 2.5.1/CUDA 12.4; validated on A100 |

## GPU runtime: Docker

Install Docker with NVIDIA Container Toolkit on the host. Run these commands
from the repository root **on the host**:

```bash
mkdir -p checkpoint results
docker build -f scripts/docker/Dockerfile -t eva:local .
docker run --rm -it --gpus device=0 --name eva-repro \
  -v "$PWD":/eva -w /eva eva:local bash
```

Inside the container, follow the [GPU benchmark](REPRODUCTION.md#gpu-run-the-complete-milena-benchmark)
or [training example](REPRODUCTION.md#training-and-fine-tuning). `cuda:0` refers
to the GPU visible inside the container. Use `exit` to return to the host.
Each output command needs a new directory; existing results are never overwritten.

The recipe uses Python 3.11, PyTorch 2.5.1, CUDA 12.4, Transformers 4.55.0,
MegaBlocks 0.7.0 and grouped-GEMM 0.1.6. It compiles the required MoE extensions;
the first build may take 30–60 minutes, depending on hardware and networking.
Attention uses PyTorch SDPA and does not require a separate FlashAttention package.
The exact dependency pins are in `scripts/docker/requirements.txt`; the validated image
and base-image digest are recorded in [validation status](RELEASE_STATUS.md).
The recipe supports CUDA build targets 7.5, 8.0 and 9.0, but that build setting
is not a claim of runtime testing on every GPU architecture.

If the default Python package index is unreachable, explicitly select a mirror:

```bash
docker build --build-arg PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple \
  -f scripts/docker/Dockerfile -t eva:local .
```

Package versions stay pinned. Record the chosen index with the build log.

## Source installation and CPU development

From the repository root, use a Python 3.10 or 3.11 virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[benchmark,design,dev,notebook]'
python -m pip check
python -m pytest -q tests
```

The CPU test suite does not require the compiled MoE extensions. For inference,
use the Docker runtime or install its exact compiled dependencies in a compatible
CUDA environment. A successful `--help` command checks installation only.

The lightweight install `python -m pip install -e .` supplies the main package.
Extras add benchmark downloads (`benchmark`), ViennaRNA (`design`), plotting and
notebooks (`notebook`), or development checks (`dev`). No model weights are
included in the wheel. See [resources](RESOURCES.md) for checkpoint downloads.

## Singularity / Apptainer (HPC clusters)


HPC users usually cannot run Docker. Convert the image with the definition
file on any machine that has Docker:

```bash
docker build -f scripts/docker/Dockerfile -t eva:local .
docker tag eva:local eva:latest  # scripts/docker/EVA.def expects this local tag
singularity build eva_latest.sif scripts/docker/EVA.def
```

The [legacy image deposit](RESOURCES.md#runtime-image) is not the current
validated source build. For this revision, build the local Docker image above
and convert it with the supplied definition file.

Always launch with `--nv` so the host NVIDIA driver is available:

```bash
singularity shell --nv eva_latest.sif
singularity exec --nv --bind /path/to/checkpoint:/eva/checkpoint \
    eva_latest.sif python /eva/tools/generate.py --help
```

Verify the runtime before running notebooks or training:

```bash
singularity exec --cleanenv --nv eva_latest.sif \
    python -c "from triton.runtime import driver; print(driver.active)"
```

### Troubleshooting: Triton `undefined symbol: cuModuleGetFunction`

Triton 3.1.0 compiles `cuda_utils.so` on first use and links it against
`-lcuda` at link time, resolving through the ldconfig cache rather than
`LD_LIBRARY_PATH`. Images built via `docker commit` on GPU hosts contain
0-byte `libcuda.so.*` placeholders: Docker `--gpus` mounts the real driver
over them (masking the bug), while Singularity `--nv` leaves them visible,
so the linker binds against an empty file and every Triton import fails
with `undefined symbol: cuModuleGetFunction`.

Images built from the current repository fix this at the source: the
placeholders are removed at build time, and `eva/_runtime_env.py` points
Triton at the injected driver via `TRITON_LIBCUDA_PATH` on every
`import eva`. Users of images released before this fix must rebuild from
the current repository.

Verify after any launch method:

```bash
python -c "from triton.runtime import driver; print(driver.active)"
```

### Pre-release smoke test

When validating an image for an HPC environment, test both launch methods.
The published validation record covers Docker; verify Singularity/Apptainer
on your target cluster before relying on it:

```bash
scripts/docker/smoke_test.sh docker eva:local
scripts/docker/smoke_test.sh singularity eva_latest.sif
```
