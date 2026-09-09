# EVA: A Long-Context Generative Foundation Model for Versatile RNA Design

<div align="center">
  <img src="fig/github_logo.svg" alt="EVA" width="800">
</div>

[Paper](https://www.biorxiv.org/content/10.64898/2026.03.17.712398v1) ·
[Model weights](https://huggingface.co/GENTEL-Lab/EVA) ·
[OpenRNA data](https://huggingface.co/datasets/GENTEL-Lab/OpenRNA-v1-114M) ·
[Website](https://evabio.net/) · [Apache-2.0 license](LICENSE)

EVA is a generative RNA foundation model for sequence scoring and conditional
sequence modeling. The released model family includes a 1.4B-parameter
mixture-of-experts model with an 8,192-token context window, trained on OpenRNA v1.
This repository contains model code, command-line tools and paper reproduction
workflows. Model weights and large datasets are downloaded separately.

## Start here

| I want to… | Start with | Requirements |
|---|---|---|
| Check a bundled result without a GPU | [CPU result recalculation](docs/REPRODUCTION.md#cpu-recalculate-stored-predictions) | Python 3.10/3.11 only |
| Run a benchmark from model weights | [135-sequence Milena example](docs/REPRODUCTION.md#gpu-run-the-complete-milena-benchmark) | Linux/NVIDIA GPU; Docker recommended |
| Use scoring or generation | [CLI guide](docs/USAGE.md) | Compatible GPU runtime and a checkpoint |
| Train or fine-tune | [Training workflow](docs/REPRODUCTION.md#training-and-fine-tuning) | Complete source checkout and GPU runtime |
| Find a paper result or comparison model | [Paper workflows](docs/PAPER_WORKFLOWS.md) | Requirements vary by workflow |

## Quick Start

Clone the repository and recalculate the bundled Milena benchmark metric:

```bash
git clone https://github.com/GENTEL-Lab/EVA.git
cd EVA
python3 scripts/reproduce_historical_benchmark.py \
  --artifact-only --output results/milena_stored
```

This standard-library-only example writes `report.json` and
`sequence_label_audit.csv` for all 135 stored predictions. The expected Spearman
correlation is **0.8360456283218484**, and the command returns 0. It recalculates
an archived result; the [GPU example](docs/REPRODUCTION.md#gpu-run-the-complete-milena-benchmark)
performs new model inference. Use a new output directory when rerunning.

## Installation

For model inference, build and enter the runtime from the repository root:

```bash
mkdir -p checkpoint results
docker build -f docker/Dockerfile -t eva:local .
docker run --rm -it --gpus device=0 --name eva-repro \
  -v "$PWD":/eva -w /eva eva:local bash
```

Then follow the GPU example below **inside the container**. Full setup, source
installation and HPC instructions are in [Installation](docs/INSTALLATION.md).

## Reproducing the paper

The complete representative example downloads a pinned public checkpoint,
checks its files, scores all 135 Milena sequences, calculates the metric and
writes diagnostic plots:

```bash
python scripts/download_reproduction_checkpoint.py \
  --model EVA_1.4B_CLM --destination checkpoint
python scripts/reproduce_milena.py \
  --checkpoint checkpoint/EVA_1.4B_CLM --output results/milena_fresh
```

Observed new-inference Spearman: **0.8394237924835843**; archived reference:
**0.8360456283218484**. The difference remains visible in the output and is not
an execution failure. Add `--strict-reference` only to request a nonzero exit
when comparison at the stored reference precision fails. Input, dependency,
scoring and plotting failures always remain errors.

The recorded A100 run scored the assay in about **12 seconds**, with **2.94 GiB**
peak PyTorch-allocated GPU memory. These are measurements for this example,
not minimum hardware requirements or estimates for other workloads; downloads,
loading, hashing and plotting add overhead.

- [Reproduction guide](docs/REPRODUCTION.md): commands, inputs, outputs and validation scope.
- [Reviewer requirements](docs/REVIEWER_REQUIREMENTS.md): six software requirements and supporting evidence.
- [Paper workflow coverage](docs/PAPER_WORKFLOWS.md): supported entry points and missing resources.
- [Validation and release status](docs/RELEASE_STATUS.md): dated evidence and publication state.

## Model and data resources

The [resource index](docs/RESOURCES.md) links checkpoints, OpenRNA and deposited
analysis data. The representative example fixes the model revision and SHA256
checksums in its manifest. Historical competitor resources and available
execution entry points are listed in [comparison models](docs/OFFICIAL_MODEL_RESOURCES.md).

## Repository layout

| Directory | Contents |
|---|---|
| `eva/`, `tools/` | Model, tokenizer and inference CLIs |
| `training/`, `finetune/`, `config/` | Training entry points and configurations |
| `scripts/` | Portable reproduction tools and shell entry points |
| `notebooks/` | Analysis notebooks and benchmark inputs |
| `reproduction/` | Versioned inputs, historical sources and validation records |
| `docs/`, `tests/` | Guides and regression tests |

The Python wheel contains the importable model and CLI packages. Use the full
source checkout for training, notebooks and paper-reproduction inputs.
Historical sources under `reproduction/` are labeled separately from supported
entry points and may preserve old paths.

## Development and support

See [CONTRIBUTING.md](CONTRIBUTING.md) for installation, tests and contribution
instructions. [Open an issue](https://github.com/GENTEL-lab/EVA/issues/new/choose)
with your commit, environment, command and error log when reporting a problem.
CPU checks run on pull requests and updates to main; GPU validation is recorded
separately and is not implied by a CPU check.

## Citation

Software citation metadata are in [CITATION.cff](CITATION.cff). Cite the exact
Git commit used for your work and the paper linked above. The integrated
**1.2.0rc1** source is public on main; its formal release and archival code DOI
are pending. Existing data DOIs identify data deposits, not this code revision.

## License

Source code is released under [Apache-2.0](LICENSE). External datasets and
third-party checkpoints retain their own upstream terms.

## Detailed guides and previous links

The detailed reference has moved into the guides below. Existing README
section links remain available here.

<a id="why-use-eva"></a> **Why Use EVA?** → [Open guide](docs/REPRODUCTION.md)

<a id="key-modules"></a> **Key Modules** → [Open guide](docs/REPRODUCTION.md)

<a id="our-journey-with-eva-starts-here-"></a> **Our Journey with EVA Starts Here 👋** → [Open guide](docs/REPRODUCTION.md)

<a id="option-a-local-source-install"></a> **Option A: Local source install** → [Open guide](docs/REPRODUCTION.md)

<a id="option-b-docker-runtime"></a> **Option B: Docker runtime** → [Open guide](docs/REPRODUCTION.md)

<a id="option-c-singularity--apptainer-hpc-clusters"></a> **Option C: Singularity / Apptainer (HPC clusters)** → [Open guide](docs/REPRODUCTION.md)

<a id="troubleshooting-triton-undefined-symbol-cumodulegetfunction"></a> **Troubleshooting: Triton `undefined symbol: cuModuleGetFunction`** → [Open guide](docs/INSTALLATION.md#troubleshooting-triton-undefined-symbol-cumodulegetfunction)

<a id="pre-release-smoke-test"></a> **Pre-release smoke test** → [Open guide](docs/INSTALLATION.md#pre-release-smoke-test)

<a id="model-download"></a> **Model download** → [Open guide](docs/REPRODUCTION.md)

<a id="release-and-versioning"></a> **Release and Versioning** → [Open guide](docs/RELEASE_STATUS.md)

<a id="repository-paths"></a> **Repository Paths** → [Open guide](docs/REPRODUCTION.md)

<a id="condition-control"></a> **Condition Control** → [Open guide](docs/USAGE.md#condition-control)

<a id="rna-types"></a> **RNA Types** → [Open guide](docs/USAGE.md#rna-types)

<a id="specieslineage"></a> **Species/Lineage** → [Open guide](docs/USAGE.md#specieslineage)

<a id="generation"></a> **Generation** → [Open guide](docs/USAGE.md#generation)

<a id="clm"></a> **CLM** → [Open guide](docs/USAGE.md#clm)

<a id="unconditional-generation"></a> **Unconditional Generation** → [Open guide](docs/USAGE.md#unconditional-generation)

<a id="conditional-generation"></a> **Conditional Generation** → [Open guide](docs/USAGE.md#conditional-generation)

<a id="continuation-mode"></a> **Continuation Mode** → [Open guide](docs/USAGE.md#continuation-mode)

<a id="glm"></a> **GLM** → [Open guide](docs/USAGE.md#glm)

<a id="unconditional-infilling"></a> **Unconditional Infilling** → [Open guide](docs/USAGE.md#unconditional-infilling)

<a id="conditional-infilling"></a> **Conditional Infilling** → [Open guide](docs/USAGE.md#conditional-infilling)

<a id="span-parameters"></a> **Span Parameters** → [Open guide](docs/USAGE.md#span-parameters)

<a id="sampling-parameters"></a> **Sampling Parameters** → [Open guide](docs/USAGE.md#sampling-parameters)

<a id="scoring"></a> **Scoring** → [Open guide](docs/USAGE.md#scoring)

<a id="rna-mode"></a> **RNA Mode** → [Open guide](docs/USAGE.md#rna-mode)

<a id="protein-mode"></a> **Protein Mode** → [Open guide](docs/USAGE.md#protein-mode)

<a id="directed-evolution"></a> **Directed Evolution** → [Open guide](docs/USAGE.md#directed-evolution)

<a id="usage"></a> **Usage** → [Open guide](docs/USAGE.md#usage)

<a id="key-parameters"></a> **Key Parameters** → [Open guide](docs/USAGE.md#key-parameters)

<a id="batch-processing-with-yaml"></a> **Batch Processing with YAML** → [Open guide](docs/USAGE.md#batch-processing-with-yaml)

<a id="generation-config-example"></a> **Generation Config Example** → [Open guide](docs/USAGE.md#generation-config-example)

<a id="scoring-config-example"></a> **Scoring Config Example** → [Open guide](docs/USAGE.md#scoring-config-example)

<a id="running"></a> **Running** → [Open guide](docs/USAGE.md#running)

<a id="inputoutput-formats"></a> **Input/Output Formats** → [Open guide](docs/USAGE.md#inputoutput-formats)

<a id="input--fasta"></a> **Input — FASTA** → [Open guide](docs/USAGE.md#input--fasta)

<a id="output--generation-fasta"></a> **Output — Generation (FASTA)** → [Open guide](docs/USAGE.md#output--generation-fasta)

<a id="output--scoring-json"></a> **Output — Scoring (JSON)** → [Open guide](docs/USAGE.md#output--scoring-json)

<a id="output--directed-evolution-fasta"></a> **Output — Directed Evolution (FASTA)** → [Open guide](docs/USAGE.md#output--directed-evolution-fasta)

<a id="data-availability"></a> **Data Availability** → [Open guide](docs/RESOURCES.md)

<a id="model--environment"></a> **Model & Environment** → [Open guide](docs/RESOURCES.md)

<a id="experiment--benchmark-data"></a> **Experiment & Benchmark Data** → [Open guide](docs/RESOURCES.md)
