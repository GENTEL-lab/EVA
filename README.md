<a id="eva-a-long-context-generative-foundation-model-for-versatile-rna-design"></a>
<p align="center">
  <img src="fig/github_logo.svg" alt="EVA — RNA foundation model" width="100%">
</p>

<h1 align="center">EVA</h1>
<p align="center"><b>A Long-Context Generative Foundation Model for Versatile RNA Design</b></p>

<p align="center">
  <a href="https://www.biorxiv.org/content/10.64898/2026.03.17.712398v1"><img src="https://img.shields.io/badge/Paper-bioRxiv-44789F?style=flat-square" alt="Paper on bioRxiv"></a>
  <a href="https://huggingface.co/GENTEL-Lab/EVA"><img src="https://img.shields.io/badge/Models-Hugging_Face-B88649?style=flat-square" alt="Models on Hugging Face"></a>
  <a href="https://huggingface.co/datasets/GENTEL-Lab/OpenRNA-v1-114M"><img src="https://img.shields.io/badge/Dataset-OpenRNA-657E76?style=flat-square" alt="OpenRNA dataset"></a>
  <a href="https://github.com/GENTEL-lab/EVA/actions/workflows/cpu.yml"><img src="https://github.com/GENTEL-lab/EVA/actions/workflows/cpu.yml/badge.svg?branch=main" alt="CPU checks on main"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache_2.0-687585?style=flat-square" alt="Apache 2.0 license"></a>
</p>

<p align="center">
  <a href="#start-here">Get started</a> &nbsp; · &nbsp;
  <a href="#reproducing-the-paper">Reproduce the paper</a> &nbsp; · &nbsp;
  <a href="#documentation">Documentation</a> &nbsp; · &nbsp;
  <a href="https://evabio.net/">Website</a> &nbsp; · &nbsp;
  <a href="#citation">Cite EVA</a>
</p>

EVA brings RNA sequence scoring and conditional sequence modeling into one
framework, with public model code, checkpoints and selected paper reproduction
workflows. Model weights and large datasets are downloaded separately.

<a id="why-use-eva"></a>
<p align="center">
  <img src="fig/readme_overview.svg" alt="1.4B-parameter MoE flagship model; 8,192-token context window; trained on OpenRNA v1" width="100%">
</p>

<a id="start-here"></a><a id="our-journey-with-eva-starts-here-"></a>

## Get started

<table>
  <tr>
    <td width="33%" valign="top">
      <h3>01 &nbsp; Explore on CPU</h3>
      <p>Recalculate a bundled benchmark result from saved predictions.</p>
      <p><sub>Python 3.10 / 3.11 · No model download</sub></p>
      <a href="#quick-start"><b>Try the quick start →</b></a>
    </td>
    <td width="34%" valign="top">
      <h3>02 &nbsp; Run on GPU</h3>
      <p>Score all 135 Milena sequences from a pinned EVA checkpoint.</p>
      <p><sub>Linux / NVIDIA GPU · Docker runtime</sub></p>
      <a href="docs/REPRODUCTION.md#gpu-run-the-complete-milena-benchmark"><b>Run the benchmark →</b></a>
    </td>
    <td width="33%" valign="top">
      <h3>03 &nbsp; Train &amp; fine-tune</h3>
      <p>Start with a small training and checkpoint save/reload workflow.</p>
      <p><sub>Source checkout · Compatible GPU runtime</sub></p>
      <a href="docs/REPRODUCTION.md#training-and-fine-tuning"><b>Explore training →</b></a>
    </td>
  </tr>
</table>

### Quick start

A first result with Python's standard library:

```bash
git clone https://github.com/GENTEL-Lab/EVA.git
cd EVA
python3 scripts/reproduce_historical_benchmark.py \
  --artifact-only --output results/milena_stored
```

**Expected:** 135 stored predictions, Spearman **0.8360456283218484**, exit code `0`.
The output directory contains `report.json` and `sequence_label_audit.csv`.
This recalculates an archived result; the GPU path below runs new inference.
Use a new output directory when rerunning.

<a id="reproducing-the-paper"></a>

## Reproduce the paper

The representative workflow connects **pinned checkpoint → 135 input sequences
→ new predictions → metrics & plots**. Inputs, model files and the scoring
protocol are bound to version and checksum records.

<a id="installation"></a>
<a id="option-a-local-source-install"></a><a id="option-b-docker-runtime"></a><a id="option-c-singularity--apptainer-hpc-clusters"></a><a id="troubleshooting-triton-undefined-symbol-cumodulegetfunction"></a><a id="pre-release-smoke-test"></a>
<details>
<summary><b>Run the GPU example</b> · Environment setup and complete inference command</summary>

From the repository root, build and enter the runtime **on the host**:

```bash
mkdir -p checkpoint results
docker build -f docker/Dockerfile -t eva:local .
docker run --rm -it --gpus device=0 --name eva-repro \
  -v "$PWD":/eva -w /eva eva:local bash
```

Then run **inside the container**:

```bash
python scripts/download_reproduction_checkpoint.py \
  --model EVA_1.4B_CLM --destination checkpoint
python scripts/reproduce_milena.py \
  --checkpoint checkpoint/EVA_1.4B_CLM --output results/milena_fresh
```

See [Installation](docs/INSTALLATION.md) for source installs, runtime versions
and Singularity / Apptainer instructions.

</details>

<details>
<summary><b>Expected results & measured runtime</b> · Reference comparison, exit codes and hardware</summary>

| Measurement | Recorded result |
|---|---|
| New-inference Spearman | **0.8394237924835843** |
| Archived reference | **0.8360456283218484** |
| Scoring time | About **12 seconds** for all 135 sequences |
| Peak PyTorch-allocated GPU memory | **2.94 GiB** |
| Environment | One NVIDIA A100-SXM4-80GB; batch size 1; pinned Docker runtime |

Default mode returns `0` when inference, metrics and plots complete successfully.
The reference difference remains visible. Add `--strict-reference` to return `2`
when comparison at the stored reference precision fails. Input, dependency,
scoring and plotting failures remain errors in either mode.

These are measurements for this example, not minimum hardware requirements
or estimates for other workloads. Downloads, loading, hashing and plotting
add overhead. A successful example does not establish reproduction of every
paper experiment. See [dated validation records](docs/RELEASE_STATUS.md).

</details>

[Protocol & outputs](docs/REPRODUCTION.md) &nbsp; · &nbsp;
[Paper workflow coverage](docs/PAPER_WORKFLOWS.md) &nbsp; · &nbsp;
[Reviewer requirements](docs/REVIEWER_REQUIREMENTS.md)

<a id="detailed-guides-and-previous-links"></a>

## Documentation

<table>
  <tr><th width="25%" align="left">Guide</th><th align="left">What you will find</th></tr>
  <tr>
    <td><a id="condition-control"></a><a id="rna-types"></a><a id="specieslineage"></a><a id="generation"></a><a id="clm"></a><a id="unconditional-generation"></a><a id="conditional-generation"></a><a id="continuation-mode"></a><a id="glm"></a><a id="unconditional-infilling"></a><a id="conditional-infilling"></a><a id="span-parameters"></a><a id="sampling-parameters"></a><a id="scoring"></a><a id="rna-mode"></a><a id="protein-mode"></a><a id="directed-evolution"></a><a id="usage"></a><a id="key-parameters"></a><a id="batch-processing-with-yaml"></a><a id="generation-config-example"></a><a id="scoring-config-example"></a><a id="running"></a><a id="inputoutput-formats"></a><a id="input--fasta"></a><a id="output--generation-fasta"></a><a id="output--scoring-json"></a><a id="output--directed-evolution-fasta"></a><a href="docs/USAGE.md"><b>Using EVA →</b></a></td>
    <td>Scoring and generation, conditioning, batch configuration, and input/output formats.</td>
  </tr>
  <tr>
    <td><a href="docs/REPRODUCTION.md"><b>Reproduction →</b></a></td>
    <td>CPU recalculation, GPU inference, training, and the scope of verified paper workflows.</td>
  </tr>
  <tr>
    <td><a id="model-and-data-resources"></a><a id="model-download"></a><a id="data-availability"></a><a id="model--environment"></a><a id="experiment--benchmark-data"></a><a href="docs/RESOURCES.md"><b>Models &amp; data →</b></a></td>
    <td>EVA checkpoints, OpenRNA, deposited analysis data, and historical comparison-model resources.</td>
  </tr>
  <tr>
    <td><a id="development-and-support"></a><a href="CONTRIBUTING.md"><b>Development →</b></a></td>
    <td>Installation for contributors, tests, pull requests, and reproducible bug reports.</td>
  </tr>
</table>

<a id="repository-layout"></a><a id="repository-paths"></a><a id="key-modules"></a>
<details>
<summary><b>Repository map</b> · Where the code and reproduction assets live</summary>

| Directory | Contents |
|---|---|
| `eva/`, `tools/` | Model, tokenizer and inference CLIs |
| `training/`, `finetune/`, `config/` | Training entry points and configurations |
| `scripts/` | Command-line workflows and automation |
| [`notebooks/`](notebooks/README.md) | Interactive analysis, plots and supporting data |
| [`reproduction/`](reproduction/README.md) | Fixed inputs, protocols, reference outputs and historical code |
| `docs/`, `tests/` | Guides and regression tests |

Notebooks and command-line workflows share the reproduction resources.
The wheel contains the importable model and CLI packages; use the complete
source checkout for training, notebooks and paper-reproduction inputs.

</details>

<a id="release-and-versioning"></a>

## Citation

Cite the paper and the exact Git commit used in your work. Software citation
metadata are available in [CITATION.cff](CITATION.cff).

The **1.2.0rc1** source is public on main; its formal release and archival code
DOI are pending. Existing data DOIs identify data deposits. See
[validation & release status](docs/RELEASE_STATUS.md) for the current scope.

<a id="license"></a>

---

<p align="center">
  <a href="LICENSE">Apache-2.0 license</a> &nbsp; · &nbsp;
  <a href="CONTRIBUTING.md">Contribute</a> &nbsp; · &nbsp;
  <a href="https://github.com/GENTEL-lab/EVA/issues/new/choose">Report an issue</a> &nbsp; · &nbsp;
  <a href="https://evabio.net/">EVA website</a>
</p>
<p align="center"><sub>External datasets and third-party checkpoints retain their upstream terms.</sub></p>
